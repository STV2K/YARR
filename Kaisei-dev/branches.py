#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import numpy as np
import torch.nn as nn
import torch.nn.functional as nnfunc
import torch
import torchvision as tv

import config
import models
from models import BidirectionalLSTM
from layers import *


class DetectionBranch(nn.Module):
    """
    EAST-like detection branch.
    """

    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.conv1_1 = nn.Conv2d(64, 1, kernel_size=1, stride=1,
                                 padding=0, bias=False)
        self.conv1_4 = nn.Conv2d(64, 4, kernel_size=1, stride=1,
                                 padding=0, bias=False)

    def forward(self, x):
        """
        :param x: output tensor from feature sharing block, expect channel_num to be 256
        :return: detection branch output
        """
        x = self.conv3(x)
        # Sigmoid are used to limit the output to 0-1
        score_map = nnfunc.sigmoid(self.conv1_1(x))
        # Angle are limited to [-45, 45]; for vertical ones, the angle is 0
        angle_map = (nnfunc.sigmoid(self.conv1_1(x)) - 0.5) * np.pi / 2
        geo_map = nnfunc.sigmoid(self.conv1_4(x)) * config.text_scale

        geometry_map = torch.cat((geo_map, angle_map), dim=1)

        return score_map, geometry_map


class RecognitionBranch(nn.Module):
    """
    FOTS/CRNN-like recognition branch.
    We change the conv-bn-relu part to FOTS-like since we adopts shared features
    while using two BiLSTM like CRNN instead of one BiLSTM and one FC of FOTS.
    """
    def __init__(self, img_h, nc, nclass, nh, n_rnn=2, leaky_relu=False):
        super().__init__()
        assert img_h % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        # nm = [64, 128, 256, 256, 512, 512, 512]
        nm = [64, 64, 128, 128, 256, 256]

        cnn = nn.Sequential()

        def conv_bn_relu(i, batch_normalization=True):
            n_in = nc if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if batch_normalization:
                cnn.add_module('bn{0}'.format(i), nn.BatchNorm2d(n_out))
            if leaky_relu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # conv_bn_relu(0)
        # cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        # conv_bn_relu(1)
        # cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        # conv_bn_relu(2, True)
        # conv_bn_relu(3)
        # cnn.add_module('pooling{0}'.format(2),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        # conv_bn_relu(4, True)
        # conv_bn_relu(5)
        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        # conv_bn_relu(6, True)  # 512x1x16
        conv_bn_relu(0)
        conv_bn_relu(1)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2, 1), (2, 1)))  # 64x16x64 todo: calc these
        conv_bn_relu(2)
        conv_bn_relu(3)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2, 1), (2, 1)))  # 128x8x32
        conv_bn_relu(4)
        conv_bn_relu(5)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 1), (2, 1)))  # 256x4x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, inp):
        # conv features
        conv = self.cnn(inp)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


class Kaisei(nn.Module):
    """
    The complete KAISEI net.
    """

    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50_block()
        self.deconv = models.deconv_block()  # TODO: consider add dropouts here if nn are struggling to converge
        self.detect = DetectionBranch()

    def forward(self, x):
        """
        :param x: output tensor from feature sharing block, expect channel_num to be 256
        :return: total branch output
        """
        x = self.resnet(x)
        residual_layers = self.resnet.residual_layers
        x = self.deconv(x, residual_layers)
        score_map, geometry_map = self.detect(x)
        # Reset residual cache
        self.resnet.residual_layers = []

        return score_map, geometry_map
