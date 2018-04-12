#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

# import math
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def bilinear_upsampling_2x():
    return nn.Upsample(scale_factor=2, mode='bilinear')


