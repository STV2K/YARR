#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import models
from PIL import Image
import numpy as np
import torchvision as tv
from torch.autograd import Variable

__image_path__ = "../testcases/SP_LOFI_mmexport1519399281474_WX.jpg"


def res50block_test():
    # block = models.resnet50(False)
    block = models.resnet50_block()
    im = Image.open(__image_path__)
    im_tensor = tv.transforms.ToTensor()(im).view(1, 3, im.size[0], -1)
    return block.forward(Variable(im_tensor))


def feature_sharing_block_test():
    im = Image.open(__image_path__)
    im_tensor = tv.transforms.ToTensor()(im)
    im_tensor = im_tensor.view(1, 3, im.size[0], -1)
    out = models.feature_sharing_block(Variable(im_tensor))
    return out


def main():
    # print(res50block_test().size())
    print(feature_sharing_block_test().size())


if __name__ == '__main__':
    main()
