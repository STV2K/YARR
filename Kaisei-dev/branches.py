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
from layers import *


class DetectionBranch(nn.Module):
    """
    EAST-like Detection Branch.
    """

    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
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


class Kaisei(nn.Module):
    """
    The complete KAISEI net.
    """

    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50_block()
        self.deconv = models.deconv_block()
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

        return score_map, geometry_map
