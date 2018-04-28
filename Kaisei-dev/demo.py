#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import os
import time

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image

import lanms
import config
import branches
import data_util as du

kaisei_det = branches.Kaisei()
if config.on_cuda:
    torch.cuda.set_device(config.gpu_list[0])
    cudnn.benchmark = config.on_cuda
    cudnn.enabled = config.on_cuda
    kaisei_det.cuda()


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
    print(score_map, geo_map.shape)
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    print(score_map, geo_map.shape)  # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = du.restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
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

    # here we filter some low score boxes by the average score map, this is different from the original paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def detect_image(net_path, img_path):
    load_net(net_path)
    for im_fn in img_path:
        im = cv2.imread(im_fn)[:, :, ::-1]
        img = Image.open(im_fn)
        start_time = time.time()
        img_resized, (ratio_w, ratio_h) = du.resize_image_fixed_square(img)
        img_tensor = transforms.ToTensor()(img_resized).unsqueeze_(0)
        if config.on_cuda:
            img_tensor = img_tensor.cuda()
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()
        score, geo = kaisei_det.forward(Variable(img_tensor))
        score = score.data.cpu().numpy()
        geo = geo.data.cpu().numpy()
        timer['net'] = time.time() - start
        boxes, timer = detect(score, geo, timer)
        print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            im_fn, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        print('[timing] {}'.format(duration))

        # save to file
        if boxes is not None:
            res_file = os.path.join(config.detect_output_dir,
                                    '{}.txt'.format(
                                        os.path.basename(im_fn).split('.')[0]))
            with open(res_file, 'w') as f:
                for box in boxes:
                    # to avoid submitting errors
                    box = du.sort_poly_points(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    ))
                    # TODO: check the geos here
                    # nbox = box.copy()
                    # nbox[0, 0] = box[0, 1]
                    # nbox[0, 1] = box[0, 0]
                    # nbox[1, 0] = box[1, 1]
                    # nbox[1, 1] = box[1, 0]
                    # nbox[2, 0] = box[2, 1]
                    # nbox[2, 1] = box[2, 0]
                    # nbox[3, 0] = box[3, 1]
                    # nbox[3, 1] = box[3, 0]

                    cv2.polylines(im[:, :, ::-1], [nbox.astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(0, 255, 255),
                                  thickness=2)
                    img_path = os.path.join(config.detect_output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])


def load_net(net_path):
    print("Loading net from ", net_path)
    kaisei_det.load_state_dict(torch.load(net_path))


if __name__ == "__main__":
    img_list = du.get_image_list(config.demo_data_path)
    print(img_list)
    detect_image(config.ckpt_path, img_list)