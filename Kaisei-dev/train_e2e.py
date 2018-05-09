#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import os
import random
import argparse

import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
# import tensorboardX

import eval
import config_e2e as config
import models
import helpers
import branches
import rec_utils
import data_util
from warpctc_pytorch import CTCLoss

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
                                      is_cuda=config.on_cuda, with_text=True)
test_loader = data_util.DataProvider(batch_size=config.test_batch_size,
                                     data_path=config.test_data_path_pami,
                                     is_cuda=config.on_cuda, with_text=True,
                                     test_set=True)

alphabet = data_util.load_alphabet_txt(config.alphabet_filepath, "GBK")
ignore_char = data_util.load_alphabet_txt(config.ignore_char_filepath, "GBK")
replace_dict = config.replace_dict
n_class = len(alphabet) + 1
converter = rec_utils.StrLabelConverter(alphabet)
criterion = CTCLoss()


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


hokuto = branches.Hokuto(n_class, config.num_hidden_state)
hokuto.apply(weights_init)

if config.continue_train:
    logger.tee('loading pretrained model from %s' % config.ckpt_path)
    hokuto.load_state_dict(torch.load(config.ckpt_path))
# print(hokuto)

if config.on_cuda:
    hokuto.cuda()
#    hokuto = torch.nn.DataParallel(hokuto, device_ids=config.gpu_list)

# loss averages
loss_det_avg = eval.LossAverage()
loss_rec_avg = eval.LossAverage()
loss_avg = eval.LossAverage()

# setup optimizer
if config.adam:
    optimizer = optim.Adam(hokuto.parameters(), lr=config.lr,
                           betas=(config.beta1, 0.999))
elif config.adadelta:
    optimizer = optim.Adadelta(hokuto.parameters(), lr=config.lr)
else:
    optimizer = optim.RMSprop(hokuto.parameters(), lr=config.lr)


# image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
# text = torch.IntTensor(opt.batchSize * 5)
# length = torch.IntTensor(opt.batchSize)


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
    n_correct = 0

    # for i in range(max(len(data_loader.data_iter), max_iter)): TODO: Check here
    for i in range(max_iter):
        data = data_loader.next()
        img_batch, score_maps, geo_maps, training_masks = data
        # img_batch = data_util.image_normalize(img_batch, config.STV2K_train_image_channel_means)
        img_batch = Variable(img_batch)
        score_maps = Variable(score_maps)
        geo_maps = Variable(geo_maps)
        training_masks = Variable(training_masks)
        pred_scores, pred_geos = net(img_batch)
        batch_loss = eval.loss(score_maps, pred_scores, geo_maps, pred_geos, training_masks)
        batch_loss = batch_loss / config.batch_size
        loss_det_avg.add(batch_loss)

    logger.tee('Test loss: %f' % (loss_avg.val()))
    loss_avg.reset()
    # i = 0
    #
    # loss_avg = eval.LossAverage
    #
    # max_iter = min(max_iter, len(data_loader))
    # for i in range(max_iter):
    #     data = val_iter.next()
    #     i += 1
    #     cpu_images, cpu_texts = data
    #     batch_size = cpu_images.size(0)
    #     utils.loadData(image, cpu_images)
    #     t, l = converter.encode(cpu_texts)
    #     utils.loadData(text, t)
    #     utils.loadData(length, l)
    #
    #     preds = crnn(image)
    #     preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    #     cost = criterion(preds, text, preds_size, length) / batch_size
    #     loss_avg.add(cost)
    #
    #     _, preds = preds.max(2)
    #     preds = preds.squeeze(2)
    #     preds = preds.transpose(1, 0).contiguous().view(-1)
    #     sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    #     for pred, target in zip(sim_preds, cpu_texts):
    #         if pred == target.lower():
    #             n_correct += 1
    #
    # raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    # for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
    #     print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    #
    # accuracy = n_correct / float(max_iter * opt.batchSize)
    # print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def train_batch(net, criterion, optimizer):
    data = train_loader.next()
    img_batch, score_maps, geo_maps, training_masks, quads, angles, contents = data
    img_batch = Variable(img_batch)
    score_maps = Variable(score_maps)
    geo_maps = Variable(geo_maps)
    training_masks = Variable(training_masks)
    ran_quads, ran_angles, ran_contents, ran_indexes = get_random_rec_datas(quads, angles, contents)
    pred_scores, pred_geos = hokuto(img_batch, ran_quads, ran_angles, ran_contents, ran_indexes)
    batch_loss = eval.loss(score_maps, pred_scores, geo_maps, pred_geos, training_masks)
    batch_loss = batch_loss / config.batch_size
    hokuto.zero_grad()
    batch_loss.backward()
    optimizer.step()
    # cpu_images, cpu_texts = data
    # batch_size = cpu_images.size(0)
    # utils.loadData(image, cpu_images)
    # t, l = converter.encode(cpu_texts)
    # utils.loadData(text, t)
    # utils.loadData(length, l)
    #
    # preds = crnn(image)
    # preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    # cost = criterion(preds, text, preds_size, length) / batch_size
    # crnn.zero_grad()
    # cost.backward()
    # optimizer.step()
    return batch_loss


def get_random_rec_datas(quads, angles, contents, batch_size = config.max_rec_batch):
    random_quads = []
    random_angles = []
    random_contents = []
    random_indexes = []

    flat_quads = []
    flat_angles = []
    flat_contents = []
    flat_indexes = []
    for i in len(quads):
        for j in len(quads[i]):
            flat_quads.append(quads[i][j])
            flat_angles.append(angles[i][j])
            content = data_util.filter_groundtruth_text(contents[i][j], ignore_char)
            flat_contents.append(content)
            flat_indexes.append(i)

    if batch_size >= len(flat_quads):
        random_quads = flat_quads
        random_angles = flat_angles
        random_contents = flat_contents
        random_indexes = flat_indexes
    else:
        ind = 0
        ind_table = list(len(flat_quads))
        while ind < batch_size:
            random_index = np.random.choice(ind_table)
            ind_table.remove(random_index)
            random_quads.append(flat_quads[random_index])
            random_angles.append(flat_angles[random_index])
            random_contents.append(flat_contents[random_index])
            random_indexes.append(flat_indexes[random_index])
            ind += 1

    return random_quads, random_angles, random_contents, random_indexes


def hokuto_train():
    i = 0
    if config.on_cuda:
        logger.tee("Using CUDA device %s id %d" % (torch.cuda.get_device_name(torch.cuda.current_device()),
                                                   torch.cuda.current_device()))
    else:
        logger.tee("CUDA disabled")
    for epoch in range(config.epoch_num):
        epoch_now = train_loader.epoch
        while epoch_now == train_loader.epoch:
            for p in hokuto.parameters():
                p.requires_grad = True
            epoch_now = train_loader.epoch
            hokuto.train()
            cost = train_batch(hokuto, criterion, optimizer)
            loss_avg.add(cost)
            i += 1

            if i % config.notify_interval == 0:
                logger.tee('[%d/%d][It-%d] Loss: %f' %
                           (train_loader.epoch, config.epoch_num, i, loss_avg.val()))
                loss_avg.reset()

            if i % config.val_interval == 0:
                val(hokuto, test_loader, criterion)

            # checkpoint
            if i % config.ckpt_interval == 0:
                try:
                    os.mkdir(config.expr_name)
                except FileExistsError:
                    pass
                torch.save(hokuto.state_dict(), '{0}/netHOKUTO_{1}_{2}.pth'.format(config.expr_name, epoch, i))


if __name__ == "__main__":
    hokuto_train()
