#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision as tv
from torch.autograd import Variable

import config
import config_e2e
import models
from models import BidirectionalLSTM
from layers import *

from PIL import Image


class DetectionBranch(nn.Module):
    """
    EAST-like detection branch.
    """

    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.conv1_1_0 = nn.Conv2d(64, 1, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.conv1_1_1 = nn.Conv2d(64, 1, kernel_size=1, stride=1,
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
        score_map = F.sigmoid(self.conv1_1_0(x))
        # Angle are limited to [-45, 45]; for vertical ones, the angle is 0
        angle_map = (F.sigmoid(self.conv1_1_1(x)) - 0.5) * np.pi / 2
        geo_map = F.sigmoid(self.conv1_4(x)) * config.text_scale

        geometry_map = torch.cat((geo_map, angle_map), dim=1)

        return score_map, geometry_map


class RecognitionBranch(nn.Module):
    """
    FOTS/CRNN-like recognition branch.
    We change the conv-bn-relu part to FOTS-like since we adopts shared features
    while using two BiLSTM like CRNN instead of one BiLSTM and one FC of FOTS.
    """
    def __init__(self, n_class, nh, n_channel=config_e2e.n_channel,
                 img_h=config_e2e.input_height, leaky_relu=False):
        super().__init__()
        assert img_h % 8 == 0, 'imgH has to be a multiple of 8'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        # nm = [64, 128, 256, 256, 512, 512, 512]
        nm = [64, 64, 128, 128, 256, 256]

        cnn = nn.Sequential()

        def conv_bn_relu(i, batch_normalization=True):
            n_in = n_channel if i == 0 else nm[i - 1]
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
            BidirectionalLSTM(256, nh, nh),
            BidirectionalLSTM(nh, nh, n_class))

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
    TODO: Consider dropout since Kaisei seems to be over-fitting.
    """

    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50_block()
        self.deconv = models.deconv_block()
        self.detect = DetectionBranch()

        self.x = 0

    def forward(self, x):
        """
        :param x: output tensor from feature sharing block, expect channel_num to be 256
        :return: total branch output
        """
        x = self.resnet(x)
        residual_layers = self.resnet.residual_layers
        x = self.deconv(x, residual_layers)
        score_map, geometry_map = self.detect(x)
        self.x = x
        # Reset residual cache
        self.resnet.residual_layers = []

        return score_map, geometry_map


class Hokuto(nn.Module):
    """
    TODO: The end-to-end framework
    """

    def __init__(self, n_class, n_hidden_status):
        super().__init__()
        # self.resnet = models.resnet50_block()
        # self.deconv = models.deconv_block()
        # self.detect = DetectionBranch()
        self.detect = Kaisei()
        self.recong = RecognitionBranch(n_class, n_hidden_status, n_channel=config_e2e.n_channel)

    def forward(self, x, quads, angles, contents, indexes, rec_flag):
        """
        :param x: output tensor from feature sharing block, expect channel_num to be 256. Batch * Channel * Height * Width
        :param quads: valid quads in the gt. Max_rec_batch * 4 * 2
        :param contents: valid contents in the gt
        :return: total branch output
        """
        rec_out = None
        # test crop and affine
        #ori_imgs = x.data
        #i = 0

        # x = self.resnet(x)
        # residual_layers = self.resnet.residual_layers
        # x = self.deconv(x, residual_layers)
        score_map, geometry_map = self.detect(x)
        x = self.detect.x
        # Reset residual cache
        # self.resnet.residual_layers = []

        if rec_flag:
            features = []
            for q, radian, index in zip(quads, angles, indexes):
                q = np.array(q)
                q = q / 4.
                q = np.clip(q, 0., 160.)
                feature = self.crop_tensor(x[index].data, int(min(q[:, 0])), int(max(q[:, 0])), int(min(q[:, 1])), int(max(q[:, 1])))
                #feature = self.crop_tensor(ori_imgs[index], int(min(q[:, 0])), int(max(q[:, 0])), int(min(q[:, 1])), int(max(q[:, 1])))
                #ori_img = feature.permute(1, 2, 0)
                #img = Image.fromarray(np.uint8(255 * np.array(ori_img)))
                #img.save('/home/hcxiao/test_affine/-angle_ori_%d.jpg' % i)

                feature_channel, feature_height, feature_width = feature.shape
                aff_width = int(feature_width * (config_e2e.input_height / feature_height))
                if aff_width <= 0 :
                    aff_width = 1

                zero_pad_flag = True
                if aff_width > config_e2e.input_max_width:
                    aff_width = config_e2e.input_max_width
                    zero_pad_flag = False

                # store angle from data_util
                aff_matrix = self.generate_affine_matrix(-radian)
                aff_flow_grid = F.affine_grid(aff_matrix, torch.Size((1, feature_channel, config_e2e.input_height, aff_width)))
                feature = feature.unsqueeze(0)
                torch.backends.cudnn.enabled = False
                feature = F.grid_sample(feature, aff_flow_grid.cuda())
                torch.backends.cudnn.enabled = True

                feature = feature[0]
                # how to determine w here?  --define in config_e2e
                # Then pad to longest width
                if zero_pad_flag:
                    zeropad = nn.ZeroPad2d((0, config_e2e.input_max_width - aff_width, 0, 0))
                    feature = zeropad(feature)
                features.append(feature)

                # test crop and affine
                #img_tensor = feature.permute(1, 2, 0)
                #print(img_tensor.size())
                #img = Image.fromarray(np.uint8(255 * np.array(img_tensor.data)))
                #img.save('/home/hcxiao/test_affine/-angle_%d.jpg' % i)
                #print('save image %d' % i)
                #i += 1

            # stack feature together
            features = torch.stack(features)
            rec_out = self.recong(features)
            return score_map, geometry_map, rec_out
        else:
            # print("No quad to rec QAQ")
            return score_map, geometry_map, []

    # The RoIAffine Operators
    @staticmethod
    def generate_affine_matrix(radian):  # , t_x, t_y): Seems we won't need t_x and t_y now.
        """
        Generate affine matrix for RoIAffine.
        Angle notation: Radian measure.
        TODO: It seems FOTS's RoIRotate is finer than ours, since it also deals with quad-to-rect problem.
              Refer to their paper for detail.
        Return a 1*2*3 tensor.
        """
        affine_m = np.zeros((2, 3))
        #radian = np.radians(degree)
        affine_m[0][0] = np.math.cos(radian)
        affine_m[0][1] = -np.math.sin(radian)
        affine_m[1][0] = np.math.sin(radian)
        affine_m[1][1] = np.math.cos(radian)
        return torch.FloatTensor(affine_m[np.newaxis, ...])

    @staticmethod
    def crop_tensor(tensor, w_min, w_max, h_min, h_max):
        """
        Crop tensor/variable with index_select.
        PyTorch Convention: nBatch * nChannel * nHeight * nWidth
        """
        # Crop the width-dim
        if w_min == w_max:
            if w_min == 0:
                w_max += 1
            else:
                w_min -= 1

        if h_min == h_max:
            if h_min == 0:
                h_max += 1
            else:
                h_min -= 1
        crop = torch.index_select(tensor, 2, torch.cuda.LongTensor(list(range(w_min, w_max))))
        crop = torch.index_select(crop, 1, torch.cuda.LongTensor(list(range(h_min, h_max))))
        return crop

    @staticmethod
    def generate_flow_grid(theta, size):
        """
        Generate flow grid according to the affine matrix for RoIAffine.
        Size convention: nBatch (1) * outHeight * outWeight * 2 (x, y), a torch.Size.
        """
        # assert len(size) == 4 and size[3] == 2 and size[1] == config.input_height
        #  and size[2] % config.input_height == 0
        # TODO: pad to longest width later; func to calc *size*
        # assert len(theta) == size[0]
        return F.affine_grid(theta, size)

    @staticmethod
    def apply_affine_grid(flow_grid, input_tensor):
        """
        Apply the affine grid.
        """
        return F.grid_sample(input_tensor, flow_grid)
