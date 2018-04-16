#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

from PIL import Image
import numpy as np
import torchvision as tv
from torch.autograd import Variable

import models
import config
import branches
import data_util


# def res50block_test():
#     # block = models.resnet50(False)
#     block = models.resnet50_block()
#     im, _ = data_util.resize_image(Image.open(config.__sample_path_1__))
#     im_tensor = tv.transforms.ToTensor()(im).view(1, 3, im.size[0], -1)
#     return block.forward(Variable(im_tensor))
#
#
# def feature_sharing_block_test():
#     im, _ = data_util.resize_image(Image.open(config.__sample_path_1__))
#     im_tensor = tv.transforms.ToTensor()(im)
#     im_tensor = im_tensor.view(1, 3, im.size[0], -1)
#     out = branches.feature_sharing_block(Variable(im_tensor))
#     return out


def detect_branch_forward_test():
    im, _ = data_util.resize_image(Image.open(config.__sample_path_1__), config.max_side_len)
    print(im.size)
    im_tensor = tv.transforms.ToTensor()(im)
    im_tensor.unsqueeze_(0)
    # Convention: [batch, channel, height, width]
    # im_tensor = im_tensor.view(1, im.size[0], im.size[1], -1)
    score, geometry = branches.Kaisei()(Variable(im_tensor))
    return score, geometry


def main():
    # print(res50block_test().size())
    s, g = detect_branch_forward_test()
    print("Score map size: " + str(s.size()))
    print(s)
    print("Geometry map size: " + str(g.size()))
    print(g)


if __name__ == '__main__':
    main()
