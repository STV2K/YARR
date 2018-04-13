#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import config
import torch
from PIL import Image


def image_normalize(images, means=[123.68 / 255, 116.78 / 255, 103.94 / 255]):
    """
    Image normalization used in EAST.
    TODO: determine the means accord to our datasets.
    TODO_DONE: Subtract with broadcasting maybe?
    """
    # chan_0 = images.narrow(1, 0, 1)
    # chan_1 = images.narrow(1, 1, 1)
    # chan_2 = images.narrow(1, 2, 1)
    # return torch.cat((chan_0 - means[0], chan_1 - means[1], chan_2 - means[2]), 1)
    return torch.Tensor.sub(images, torch.Tensor(means).view(1, 3, 1, 1))


def resize_image(image, max_side_len=1280):
    """
    Adopted from EAST code.
    resize image to a size multiple of 32 which is required by the network.
    :param image: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    w, h = image.size

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    img = image.resize((int(resize_w), int(resize_h)))  # , Image.BILINEAR)

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return img, (ratio_h, ratio_w)


if __name__ == '__main__':
        im = Image.open(config.__sample_path_1__)
        im, o = resize_image(im, 3200)
        print(im.size)
        # im.show()
