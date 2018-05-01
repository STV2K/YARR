#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import time
import cv2
import numpy as np
import torch
from torch.autograd import Variable

import lanms
import config
import helpers
import data_util as du

logger = helpers.ExpLogger(config.log_file_name + "eval")

# The following methods are adopted from EAST.
def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    """
    dice loss coefficient calculating.
    """
    eps = 1e-5
    # Custom: add negative_loss here
    negative_loss = torch.mean(y * y_true_cls.sub(1) * training_mask)
    intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    dice_loss = 1. - (2 * intersection / union) + negative_loss
    logger.tee("neg loss: " + str(negative_loss.data))
    logger.tee("cls loss: " + str(dice_loss.data))
    # TODO: check how neg_loss works; scalar to tensorboard
    # tf.summary.scalar('classification_dice_loss', loss)
    return dice_loss


def loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
    """
    define the loss used for training, containing two part,
    the first part we use dice loss instead of weighted log_loss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some invalid text
    :return:
    """
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    # print(torch.split(y_true_geo, 5, dim=1))
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, dim=1)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, dim=1)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    l_AABB = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
    l_theta = 1 - torch.cos(theta_pred - theta_gt)
    # TODO: add to scalar
    # tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    # tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    l_g = l_AABB + 20 * l_theta
    # print("Loss:", l_g, " AABB:", l_AABB, " theta:", l_theta, " ret:", torch.mean(l_g * y_true_cls * training_mask) +
    #      classification_loss)
    logger.tee("l_g loss: " + str(torch.mean(l_g * y_true_cls * training_mask).data))
    return torch.mean(l_g * y_true_cls * training_mask) + classification_loss


class LossAverage(object):
    """
    Adopted from CRNN.
    Compute average for `torch.Variable` and `torch.Tensor`.
    """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

