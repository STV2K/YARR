#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import time
import cv2
import numpy as np
import torch

import config
import lanms
import data_util as du


# The following methods are adopted from EAST.
def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    """
    dice loss coefficient calculating.
    """
    eps = 1e-5
    intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    dice_loss = 1. - (2 * intersection / union)
    # TODO: Scalar to tensorboard
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
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 5, dim=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 5, dim=3)
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

    return torch.mean(l_g * y_true_cls * training_mask) + classification_loss


def detect(score_map, geo_map, timer, score_map_thresh=config.score_map_threshold,
           box_thresh=config.box_threshold, nms_thresh=config.nms_threshold):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshold for score map
    :param box_thresh: threshold for boxes
    :param nms_thresh: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = du.restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thresh)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer
