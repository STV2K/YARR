#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import argparse
import random

import models
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import tensorboardX

import eval
import config
import branches
import data_util


# gpu_ids = list(range(len(config.gpu_list.split(','))))
# gpu_id = config.gpu_list

random_seed = random.randint(1, 2292014)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

cudnn.benchmark = True
cudnn.enabled = True

train_loader = data_util.DataProvider(batch_size=config.batch_size,
                                      data_path=config.training_data_path_pami,
                                      is_cuda=True)
test_loader = data_util.DataProvider(batch_size=config.test_batch_size,
                                     data_path=config.test_data_path_pami,
                                     is_cuda=True)


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

if config.continue_train != '':
    print('loading pretrained model from %s' % config.ckpt_path)
    kaisei.load_state_dict(torch.load(config.ckpt_path))
# print(kaisei)

if config.on_cuda:
    kaisei.cuda()
    kaisei = torch.nn.DataParallel(kaisei, device_ids=config.gpu_list)


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


# TODO: adopt this
def val(net, dataset, criterion, max_iter=100):
    """
    Adopted from CRNN.
    Detect.
    """
    print('Start val===')

    for p in kaisei.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = eval.LossAverage

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


# TODO: adopt this
def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def detection_train():
    # TODO: checkpoint, gpu_config, restoring
    input_images = []
    input_score_maps = []
    input_geo_maps = []
    input_training_masks = []
    criterion = eval.loss

    for epoch in range(config.epoch_num):
        epoch_now = train_loader.epoch
        i = 0
        while epoch_now == train_loader.epoch:
            for p in kaisei.parameters():
                p.requires_grad = True
            kaisei.train()

            cost = trainBatch(kaisei, criterion, optimizer)
            loss_avg.add(cost)
            i += 1

            if i % config.notify_interval == 0:
                print('[%d/%d][It-%d] Loss: %f' %
                      (epoch, config.epoch_num, i, loss_avg.val()))
                loss_avg.reset()

            if i % config.val_interval == 0:
                val(kaisei, test_loader, criterion)

            # checkpoint
            if i % config.ckpt_path == 0:
                torch.save(kaisei.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(config.expr_name, epoch, i))


if __name__ == "__main__":
    detection_train()
