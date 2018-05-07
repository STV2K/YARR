#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import os
import random
# import argparse

import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# import tensorboardX

import eval
import config
# import models
import helpers
import branches
import data_util


# gpu_ids = list(range(len(config.gpu_list.split(','))))
# gpu_id = config.gpu_list
logger = helpers.ExpLogger(config.log_file_name)

random_seed = random.randint(1, 2292014)
logger.tee("Random seed set to %d" % random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Assume on single GPU
torch.cuda.set_device(config.gpu_list[0])
cudnn.benchmark = config.on_cuda
cudnn.enabled = config.on_cuda

train_loader = data_util.DataProvider(batch_size=config.batch_size,
                                      data_path=config.training_data_path_pami,
                                      is_cuda=config.on_cuda)
test_loader = data_util.DataProvider(batch_size=config.test_batch_size,
                                     data_path=config.test_data_path_pami,
                                     is_cuda=config.on_cuda, test_set=True)


def weights_init(module):
    """
    Weight initialization code adopted from [CRNN](https://github.com/meijieru/crnn.pytorch).
    """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


kaisei = branches.Kaisei()
kaisei.apply(weights_init)

if config.continue_train:
    logger.tee('loading pretrained model from %s' % config.ckpt_path)
    kaisei.load_state_dict(torch.load(config.ckpt_path))
# print(kaisei)

if config.on_cuda:
    kaisei.cuda()
#    kaisei = torch.nn.DataParallel(kaisei, device_ids=config.gpu_list)

# loss average
loss_avg = eval.LossAverage()

# setup optimizer
if config.adam:
    optimizer = optim.Adam(kaisei.parameters(), lr=config.lr,
                           betas=(config.beta1, 0.999))
elif config.adadelta:
    optimizer = optim.Adadelta(kaisei.parameters(), lr=config.lr)
else:
    optimizer = optim.RMSprop(kaisei.parameters(), lr=config.lr)


def val(net, dataset, criterion, max_iter=config.test_iter_num):
    """
    Adopted from CRNN.
    Valuate.
    """
    logger.tee('Start val')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = test_loader

    # for i in range(max(len(data_loader.data_iter), max_iter)): TODO: Check here
    for i in range(max_iter):
        data = data_loader.next()
        img_batch, score_maps, geo_maps, training_masks = data
        img_batch = Variable(img_batch)
        score_maps = Variable(score_maps)
        geo_maps = Variable(geo_maps)
        training_masks = Variable(training_masks)
        pred_scores, pred_geos = net(img_batch)
        batch_loss = eval.loss(score_maps, pred_scores, geo_maps, pred_geos, training_masks)
        batch_loss = batch_loss / config.batch_size
        loss_avg.add(batch_loss)

    logger.tee('Test loss: %f' % (loss_avg.val()))
    loss_avg.reset()


def train_batch(net, criterion, optimizer):
    data = train_loader.next()
    img_batch, score_maps, geo_maps, training_masks = data
    img_batch = Variable(img_batch)
    score_maps = Variable(score_maps)
    geo_maps = Variable(geo_maps)
    training_masks = Variable(training_masks)
    pred_scores, pred_geos = kaisei(img_batch)
    batch_loss = eval.loss(score_maps, pred_scores, geo_maps, pred_geos, training_masks)
    batch_loss = batch_loss / config.batch_size
    kaisei.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss


def detection_train():
    criterion = eval.loss
    i = 0
    if config.on_cuda:
        logger.tee("Using CUDA device %s id %d" % (torch.cuda.get_device_name(torch.cuda.current_device()),
                                                   torch.cuda.current_device()))
    else:
        logger.tee("CUDA disabled")
    for epoch in range(config.epoch_num):
        epoch_now = train_loader.epoch
        while epoch_now == train_loader.epoch:
            for p in kaisei.parameters():
                p.requires_grad = True
            epoch_now = train_loader.epoch
            kaisei.train()
            cost = train_batch(kaisei, criterion, optimizer)
            loss_avg.add(cost)
            i += 1

            if i % config.notify_interval == 0:
                logger.tee('[%d/%d][It-%d] Loss: %f' %
                           (train_loader.epoch, config.epoch_num, i, loss_avg.val()))
                loss_avg.reset()

            if i % config.val_interval == 0:
                val(kaisei, test_loader, criterion)

            # checkpoint
            if i % config.ckpt_interval == 0:
                try:
                    os.mkdir(config.expr_name)
                except FileExistsError:
                    pass
                torch.save(kaisei.state_dict(), '{0}/netKAISEI_{1}_{2}.pth'.format(config.expr_name, epoch, i))


if __name__ == "__main__":
    detection_train()
