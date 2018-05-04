#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import os
import sys

import cv2
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageDraw2
from shapely.geometry import Polygon
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoaderIter

# from torch.utils.data import sampler


if sys.platform == "darwin":
    import matplotlib as mil
    mil.use('TkAgg')  # Fix RTE caused by pyplot under macOS
import matplotlib.pyplot as plt
# import matplotlib.patches as Patches

import config


# Tensor Transformation
def image_normalize(images, means=(112.80 / 255, 106.65 / 255, 101.02 / 255)):
    """
    Image normalization used in EAST.
    TODO_SKIP: calc the stds accord to our dataset(DONE) and use torchvision's transformation.
    TODO_DONE: determine the means accord to our dataset.
    EAST means: [123.68, 116.78, 103.94] -- ICDAR?
    STV2K Train means: [112.7965, 106.6500, 101.0181]
    TODO_DONE: Subtract with broadcasting maybe?
    """
    # chan_0 = images.narrow(1, 0, 1)
    # chan_1 = images.narrow(1, 1, 1)
    # chan_2 = images.narrow(1, 2, 1)
    # return torch.cat((chan_0 - means[0], chan_1 - means[1], chan_2 - means[2]), 1)
    return torch.Tensor.sub(images, torch.Tensor(means).view(1, 3, 1, 1))


# PIL Transformation
def resize_image(image, max_side_len):
    """
    Adopted from EAST code. Modified to use PIL.
    resize image to a size multiple of 32 which is required by the network.
    :param image: the original PIL image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    w, h = image.size

    # limit the max side
    if max(h, w) > max_side_len:
        ratio = float(max_side_len) / h if h > w else float(max_side_len) / w
    else:
        ratio = 1.
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = image.resize((int(resize_w), int(resize_h)))  # , Image.BILINEAR)

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_w, ratio_h)


def resize_image_fixed_square(image, fixed_len=config.fixed_len):
    """
    Resize images to square of fixed_len. We use this as a compromise for training, since batch requires tensors of same
    size. Also make sure fixed_len is a size multiple of 32 which is required by the network.
    :param image: the original PIL image
    :param fixed_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    w, h = image.size

    fixed_len = fixed_len if fixed_len % 32 == 0 else (fixed_len // 32 - 1) * 32
    im = image.resize((int(fixed_len), int(fixed_len)))  # , Image.BILINEAR)

    ratio_h = fixed_len / float(h)
    ratio_w = fixed_len / float(w)

    return im, (ratio_w, ratio_h)


def random_crop(img, quads, contents, bools, prob_bg=0.05, prob_partial=0.55,
                top_left_point_ratio=0.5, min_crop_ratio=0.3, aspect_ratio_range=(0.7, 1.3)):
    """
    Randomly cropping.
    :param img: An PIL Image obj.
    :param quads: Annotation of quadrilaterals with size and direction validated,
                  formatted as [[(x1, y1), ...], ...](np.array).
    :param contents: Annotation of text contents.
    :param bools: Bools of text contents.
    :param prob_bg: Probability to crop a background area, otherwise crop partial or don't crop.
    :param prob_partial: Probability to crop a partial out, otherwise don't crop.
    :param top_left_point_ratio: Which area top-left point of crop area will be generated into.
    :param min_crop_ratio: Minimal size ratio allowed for the cropped image.
    :param aspect_ratio_range: As the var name suggests.
    :return: Cropped img, quads and contents.
    """
    assert top_left_point_ratio + min_crop_ratio < 1
    img = img.copy()
    rand = np.random.random()
    if rand < prob_bg + prob_partial:
        imw, imh = img.size
        minw = int(imw * min_crop_ratio)
        minh = int(imh * min_crop_ratio)
        x = int(np.random.random() * top_left_point_ratio * imw)
        y = int(np.random.random() * top_left_point_ratio * imh)
        w = np.random.randint(minw, imw - x)
        minh = max(int(aspect_ratio_range[0] * w), minh)
        maxh = min(int(aspect_ratio_range[1] * w), imh - y)
        min_h = min(minh, maxh)
        max_h = max(minh, maxh)
        if min_h >= max_h:
            h = max_h
        else:
            h = np.random.randint(min_h, max_h)
        # print(h, w, minh, maxh)
        boarder_left = boarder_top = 0
        boarder_right = imw
        boarder_bottom = imh
        draw = ImageDraw.Draw(img)
        if rand < prob_bg:
            # Generate negative samples
            crop_area = Polygon([(x, y), (x, y + h), (x + w, y + h), (x + w, y)])
            for quad, content in zip(quads, contents):
                quad_area = Polygon(quad)
                if quad_area.intersects(crop_area):  # within() and crosses() seems to be inadequate
                    # fill poly
                    draw.polygon(quad, fill=(w % 256, h % 256, (w + h) % 256))
            del draw  # Not sure if this is necessary
            crop_img = img.copy().crop((x, y, w + x, h + y))
            crop_img.load()
            # print("Cropping bg ", (x, y, w + x, h + y))
            return crop_img, [], [], []
        else:
            # crop partial, may be minor bug with the shapely judgements
            crop_update = True
            extend_update = True
            out_quad = []
            out_cont = []
            out_bool = []
            for quad, content, tag_bool in zip(quads, contents, bools):
                if crop_update:
                    crop_area = Polygon([(x, y), (x, y + h), (x + w, y + h), (x + w, y)])
                    crop_update = False
                if extend_update:
                    extend_area = Polygon([(boarder_top, boarder_left), (boarder_top, boarder_right),
                                            (boarder_bottom, boarder_right), (boarder_bottom, boarder_left)])
                    extend_update = False
                # assert crop_area.within(extend_area)
                quad_area = Polygon(quad)
                if quad_area.within(crop_area):
                    # Quad in crop, add to outset
                    out_cont.append(content)
                    out_quad.append(quad)
                    out_bool.append(tag_bool)
                elif quad_area.intersects(crop_area):
                    if not quad_area.within(extend_area):
                        # Quad crosses extinct, extend and crop area, fill poly and discard
                        draw.polygon(quad, fill=(w % 256, h % 256, (w + h) % 256))
                        # print("OEC ", content)
                    else:
                        # Quad is in extend and crop area, add to and expand crop
                        # for simplicity, we won't choose to discard then fill poly
                        [(q_xmin, q_ymin), (q_xmax, q_ymax)] = quad2rect(quad)
                        if q_xmin < x:
                            w += int(x - q_ymax)
                            x = q_xmin
                        if q_ymin < y:
                            h += int(y - q_ymin)
                            y = q_ymin
                        w = int(q_xmax - x) if q_xmax > (x + w) else w
                        h = int(q_ymax - y) if q_ymax > (y + h) else h
                        crop_update = True
                        out_cont.append(content)
                        out_quad.append(quad)
                        out_bool.append(tag_bool)
                        # print("EC ", content)
                else:
                    # print("O/E ", content)
                    # if quad_area.within(extend_area) or quad_area.crosses(extend_area):
                    if quad_area.intersects(extend_area):
                        # Within:
                        # Quad is in extend area, discard then shrink extend
                        # for simplicity, we won't choose to add to and expand crop
                        # Crosses:
                        # Quad is in extinct and extend area, discard and shrink extend
                        [(q_xmin, q_ymin), (q_xmax, q_ymax)] = quad2rect(quad)
                        boarder_top = q_ymin if q_ymin > boarder_top else boarder_top
                        boarder_bottom = q_ymax if q_ymax < boarder_bottom else boarder_bottom
                        boarder_left = q_xmin if q_xmin > boarder_left else boarder_left
                        boarder_right = q_xmax if q_xmax < boarder_right else boarder_right
                        extend_update = True
                        # print("E shrink")
                    # else:  # Otherwise quad is in extinct, just discard
                        # print("Quad in extinct: ", content)
            # print("Cropping partial ", (x, y, w + x, h + y))
            del draw
            crop_img = img.copy().crop((x, y, w + x, h + y))
            crop_img.load()
            # Update quads
            if len(out_quad):
                out_quad = np.array(out_quad)
                out_quad[:, :, 0] -= x
                out_quad[:, :, 1] -= y
    else:
        # Otherwise do no cropping, return the original image and annotations
        # print("No crop")
        crop_img = img
        out_quad = quads
        out_cont = contents
        out_bool = bools

    return crop_img, out_quad, out_cont, out_bool


# Computational Geometry
# Most of these functions are adopted from EAST.
def polygon_area(poly):
    """
    compute area of a polygon
    :param poly:
    :return: area of poly
    """
    # return Polygon(poly).area
    # DONE: Time and compare
    #       Shapely costs 1.6x if Polygon obj creation costs are included
    #       However here we shall keep the edge impl for points direction checking,
    #       as shapely impl returns all positive results.
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def check_and_validate_polys(polys, tags, tag_bools, img_size):
    """
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    """
    (w, h) = img_size
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []
    validated_bools = []
    for poly, tag, tag_bool in zip(polys, tags, tag_bools):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            print('invalid poly:', poly, tag)
            continue
        if p_area > 0:
            # print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
        validated_bools.append(tag_bool)
    return np.array(validated_polys), np.array(validated_tags), np.array(validated_bools)


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1~p2
    # print("Computing " + str(p3) + " to line " + str(p1) + str(p2))
    # print("Get: " + str(np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)))
    try:
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    except:
        print("Computing " + str(p3) + " to line " + str(p1) + str(p2))


def fit_line(p1, p2):
    # fit a line ax+by+c=0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    # line1 0=ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype=np.float32)


def line_vertical(line, point):
    # get the vertical line from line across point
    if line[1] == 0:
        vertical = [0, -1, point[1]]
    else:
        if line[0] == 0:
            vertical = [1, 0, -point[0]]
        else:
            vertical = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
    return vertical


def rectangle_from_parallelogram(poly):
    """
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    """
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p0 and p2
            # p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_vertical = line_vertical(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_vertical)
            # p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_vertical = line_vertical(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_vertical)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_vertical = line_vertical(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_vertical)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_vertical = line_vertical(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_vertical)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p1 and p3
            # p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_vertical = line_vertical(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_vertical)
            # p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_vertical = line_vertical(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_vertical)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_vertical = line_vertical(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_vertical)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_vertical = line_vertical(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_vertical)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_poly_points(poly):
    min_axis = np.argmin(np.sum(poly, axis=1))
    poly = poly[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(poly[0, 0] - poly[1, 0]) > abs(poly[0, 1] - poly[1, 1]):
        return poly
    else:
        return poly[[0, 3, 2, 1]]


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点
        p_lowest_right = (p_lowest - 1) % 4
        # p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        # if angle <= 0:
            # print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            # 这个点为p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # 这个点为p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)


def generate_rbox(im_size, polys, tag_bools, tag_content, img_name=""):
    w, h = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((5, h, w), dtype=np.float32)
    # mask used during training to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    for poly_idx, poly_tag in enumerate(zip(polys, tag_bools, tag_content)):
        poly = poly_tag[0]
        tag = poly_tag[1]
        poly_content = poly_tag[2]
        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        # We don't shrink poly like EAST did to avoid computational geometry overkill
        # shrunk_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, poly.astype(np.int32)[np.newaxis, :, :], 1)
        cv2.fillPoly(poly_mask, poly.astype(np.int32)[np.newaxis, :, :], poly_idx + 1)
        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if tag:
            if min(poly_h, poly_w) < config.min_text_size:
                # print("Ignore for min_text_size: " + poly_content + "(" +
                # str(min(poly_h, poly_w)) + ")[" + str(poly_idx) + "]")
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            # elif poly_h * poly_w // len(poly_content) < config.min_char_avgsize:
                # print("Ignore for min_char_avgsize: " + poly_content + "(" +
                #       str(poly_h * poly_w // len(poly_content)) + ")[" + str(poly_idx) + "]")
                # cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        else:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        # if geometry == 'RBOX':
        # 对任意两个顶点的组合生成一个平行四边形
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            # if p2 is None:
            #    print("P2 returned None: ", img_name, poly)
            # if np.linalg.norm(p2 - p1) == 0.:
            #     print("P2-P1 returned 0: ", img_name, poly)
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # 平行线经过p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # 经过p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            # new_p0 = p0
            new_p1 = p1
            # new_p2 = p2
            # new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            # try:
            #     if new_p2 is None:
            #         print("NP2 returned None: ", img_name, poly)
            #     if np.linalg.norm(new_p2 - p0) == 0.:
            #         print("NP2-P0 returned 0: ", img_name, poly)
            # except:
            #     print("Weird occuried ", img_name, poly)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            # new_p1 = p1
            # new_p2 = p2
            # new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [abs(polygon_area(t)) for t in fitted_parallelograms]
        # areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort the polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectangle = rectangle_from_parallelogram(parallelogram)
        rectangle, rotate_angle = sort_rectangle(rectangle)

        p0_rect, p1_rect, p2_rect, p3_rect = rectangle
        # Ugly, but must exchange point axis since we use different conventions of (w, h) rather than (h, w) in EAST
        p0_rect_ex = exchange_point_axis(p0_rect)
        p1_rect_ex = exchange_point_axis(p1_rect)
        p2_rect_ex = exchange_point_axis(p2_rect)
        p3_rect_ex = exchange_point_axis(p3_rect)
        for y, x in xy_in_poly:
            point = np.array([y, x], dtype=np.float32)
            # top
            geo_map[0, y, x] = point_dist_to_line(p0_rect_ex, p1_rect_ex, point)
            # right
            geo_map[1, y, x] = point_dist_to_line(p1_rect_ex, p2_rect_ex, point)
            # down
            geo_map[2, y, x] = point_dist_to_line(p2_rect_ex, p3_rect_ex, point)
            # left
            geo_map[3, y, x] = point_dist_to_line(p3_rect_ex, p0_rect_ex, point)
            # angle
            geo_map[4, y, x] = rotate_angle
    return score_map[np.newaxis, ...], geo_map, training_mask[np.newaxis, ...]


# Dataset helpers
def get_image_list(file_path):
    file_list = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        for root, dirs, files in os.walk(file_path):
            for f in files:
                # Only check files under root
                if f.split(".")[-1] in ext and root == file_path:
                    file_list.append(os.path.join(file_path, f))
    return file_list


def load_annotation(label_path, encoding="GBK"):
    """
    Load the new STV2K label file into arrays using EAST Fashion.
    :return *text_quad*, *text_content* and *text_bool*.

    The label file ought to be encoded with GB2312 and organized as:
        [Line 1] x1, y1, x2, y2, x3, y3, x4, y4 (all integers with origin on the top left)
               (TopRight)(BtmR)  (BtmL) (TopLeft)
        [Line 2] Text contents or blank line if unrecognizable
        [Line 3] Blank line
        [Repeat for other bounding boxes]

    NOTICE: Some indices in the label files may be out of boundary, we deal with np.clip in check_and_validate_polys.
    """
    text_quad = []
    text_content = []
    text_bool = []
    with open(label_path, "r", encoding=encoding) as lf:
        count = 0
        for line in lf:
            if count % 3 == 0 and line != "":
                line = line.strip().split(",")
                x_axis = list(map(int, line[0:-1:2]))
                y_axis = list(map(int, line[1::2]))
                temp_quad = []
                # Resize quads accordingly
                for i in range(len(x_axis)):
                    # Out-of-boundary indices are dealt in check_and_validate_polys().
                    # x_axis[i] = 0 if x_axis[i] < 0 else x_axis[i]
                    # y_axis[i] = 0 if y_axis[i] < 0 else y_axis[i]
                    # x_axis[i] = image_size[0] if x_axis[i] > image_size[0] else x_axis[i]
                    # y_axis[i] = image_size[1] if y_axis[i] > image_size[1] else y_axis[i]
                    # temp_quad.append((int(x_axis[i] * im_ratio[0]), int(y_axis[i] * im_ratio[1])))
                    temp_quad.append([x_axis[i], y_axis[i]])
                text_quad.append(temp_quad)
            elif count % 3 == 1:
                content = line.strip()
                text_content.append(content)
                if len(content):
                    text_bool.append(True)
                else:
                    text_bool.append(False)
            count += 1

    return np.array(text_quad, dtype=np.float32), text_content, np.array(text_bool, dtype=np.bool)


def load_data(file_path=config.demo_data_path):
    image_list = get_image_list(file_path)
    np.random.shuffle(image_list)

    ret = []
    for img_path in image_list:
        img = Image.open(img_path)
        img_ori_size = img.size
        img_filename = str(os.path.basename(img_path))
        label_path = img_path.replace(img_filename.split('.')[1], 'txt')
        img, resize_ratio = resize_image_fixed_square(img, config.fixed_len)
        label_quad, label_content, label_bool = load_annotation(label_path)
        vali_quad, vali_cont, vali_bool = check_and_validate_polys(label_quad, label_content, label_bool, img_ori_size)
        score_map, geo_map, training_mask = generate_rbox(img.size, vali_quad, vali_bool, vali_cont)
        ret.append([img_filename, img, resize_ratio,
                    vali_quad, vali_cont, vali_bool,
                    score_map, geo_map, training_mask])
    return ret


# Visualizing
def show_gray_colormap(array):
    plt.imshow(array, cmap="gray")
    plt.show()


# Other helpers
def is_two_rect_overlapping(rect_A, rect_B):
    (xmin_A, ymin_A), (xmax_A, ymax_A) = rect_A
    (xmin_B, ymin_B), (xmax_B, ymax_B) = rect_B
    if xmin_A < xmin_B and xmax_A < xmin_B:
        return False
    elif xmin_A > xmax_B and xmax_A > xmax_B:
        return False
    if ymin_A < ymin_B and ymax_A < ymin_B:
        return False
    elif ymin_A > ymax_B and ymax_A > ymax_B:
        return False
    return True


def rect2Polygon(rect):
    [(x0, y0), (x1, y1)] = rect
    return Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])


def vis_quad(img, quads):
    im = img.copy()
    draw = ImageDraw.Draw(im)
    for quad in quads:
        draw.polygon(quad)
    del draw
    im.show()


def quad2rect(quad):
    """
    Return the rectangle corresponding to the quadrilateral.
    :param quad: A numpy array.
    :return: the rectangle.
    """
    # xmin = quad[0][0]
    # ymin = quad[0][1]
    # xmax = ymax = 0
    # for p in quad:
    #     xmin = min(p[0], xmin)
    #     xmax = max(p[0], xmax)
    #     ymin = min(p[1], ymin)
    #     ymax = max(p[1], ymax)
    # return [(xmin, ymin), (xmax, ymax)]
    return np.array([(min(quad[:, 0]), min(quad[:, 1])), (max(quad[:, 0]), max(quad[:, 1]))])


def rect2quad(rect):
    [(x0, y0), (x1, y1)] = rect
    return [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]


def load_alphabet_txt(filename, encoding="GB18030"):
    alphabet = []
    with open(filename, "r", encoding=encoding) as f:
        for line in f:
            alphabet.append(line.strip())
    return alphabet


def exchange_point_axis(p):
    return np.array([p[1], p[0]])


def alphabet_gen(paths):
    alphabet = []
    for path in paths:
        img_list = get_image_list(path)

        for img_path in img_list:
            img_filename = str(os.path.basename(img_path))
            label_path = img_path.replace(img_filename.split('.')[1], 'txt')
            _, content, __ = load_annotation(label_path)
            for line in content:
                alphabet += line

    return list(set(alphabet))


def write_alphabet(alphabet, filename, encoding="GBK"):
    with open(filename, "w", encoding=encoding) as f:
        for line in alphabet:
            f.write(line + '\n')


def calc_image_channel_mean(img_list):
    sum_up = np.array([0., 0., 0.])
    for img in tqdm(img_list):
        im_array = np.array(Image.open(img))
        # sample_sum = np.array([0, 0, 0])
        # sample_sum[0] += im_array[:, :, 0].sum()
        # sample_sum[1] += im_array[:, :, 1].sum()
        # sample_sum[2] += im_array[:, :, 2].sum()
        sum_up += np.mean(im_array, (0, 1))  # / im_array.shape[0] / im_array.shape[1]
    return sum_up / len(img_list)


class ImageChannelStdCalc:
    sample_num = 0

    def __init__(self, means):
        self.means = np.array(means)
        self.deviation_accumulate = np.array([0.] * len(means))

    def add_sample(self, sample):
        assert len(sample) == len(self.means)
        self.deviation_accumulate += np.square(sample - self.means)
        self.sample_num += 1

    def get_std(self):
        return np.sqrt(self.deviation_accumulate / self.sample_num)


def calc_image_channel_std(img_list, img_means):
    std_calc = ImageChannelStdCalc(np.array(img_means))
    for img in tqdm(img_list):
        im_array = np.array(Image.open(img))
        std_calc.add_sample(np.mean(im_array, (0, 1)))
    return std_calc.get_std()


def np_img_color_twitch(img, prob_1=0.6, prob_2=0.25, prob_3=0.10, max_twitch=0.38):
    """
    Twitch color of an image for augmentation on channel level.
    :param img: Image as np.array with convention(w, h, c)
    :param prob_1: Probability to twitch one channel
    :param prob_2: Probability to twitch two channels
    :param prob_3: Probability to twitch all channels
    :param max_twitch: Maximum twitch scale allowed.
    :return: Twitched image.
    """
    channel_idx = [0, 1, 2]
    prob = np.random.random()
    if prob < (prob_1 + prob_2 + prob_3):
        twitch_one_channel(img, channel_idx, max_twitch)
        if prob > prob_1:
            twitch_one_channel(img, channel_idx, max_twitch)
            if prob > (prob_1 + prob_2):
                twitch_one_channel(img, channel_idx, max_twitch)

    return np.clip(img, 0, 255)


def twitch_one_channel(img, channel_idx, max_twitch):
    ch = np.random.choice(channel_idx)
    scale = (np.random.random() * 2 - 1) * max_twitch
    channel_idx.remove(ch)
    if scale > 0:
        # twitch with different methods to prevent out-of-board
        img[:, :, ch] = img[:, :, ch] + scale * (255 - img[:, :, ch])
    else:
        img[:, :, ch] = img[:, :, ch] * (scale + 1)
    # print("Twitch ch ", ch, " with scale ", scale)


# PyTorch Data Warp-up
# TODO_COMPROMISED: fix issue that default collate_fn cannot deal with various sized tensor, which requires turning
#  into lists or to sample pictures for mini-batches in an aspect-ratio-respecting way
# COMPROMISE: we chose to resize all stv2k images to square resolution during training.
# def collate_fn(batch):
#     # Note that batch is a list
#     batch = list(map(list, zip(*batch)))  # transpose list of list
#     out = None
#     # You should know that batch[0] is a fixed-size tensor since you're using your customized Dataset
#     # reshape batch[0] as (N, H, W)
#     # batch[1] contains tensors of different sizes; just let it be a list.
#     # If your num_workers in DataLoader is bigger than 0
#     #     numel = sum([x.numel() for x in batch[0]])
#     #     storage = batch[0][0].storage()._new_shared(numel)
#     #     out = batch[0][0].new(storage)
#     batch[0] = torch.stack(batch[0], 0, out=out)
#     return batch


class STV2KDetDataset(Dataset):

    def __init__(self, path=config.training_data_path_z440, with_text=False, is_test_set=False):
        self.image_list = get_image_list(path)
        self.toTensor = transforms.ToTensor()
        self.data_path = path
        self.is_test_set = is_test_set
        self.with_text = with_text

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        # print("Getting img:", img_path)
        img = Image.open(img_path)
        img_ori_size = img.size
        img_filename = str(os.path.basename(img_path))
        label_path = img_path.replace(img_filename.split('.')[1], 'txt')
        # Generate score_map and geo_map of 1/4 (w, h) of image size to match the network output
        # Fix: 0~1 // 4 == 0!
        # NB: Passing 1/4 ratio to generate score and geo maps may evoke mysterious computational geometry problems.
        # ratio_1_4 = (resize_ratio[0] / 4, resize_ratio[1] / 4)
        # size_1_4 = (img.size[0] // 4, img.size[1] // 4)
        # TODO_Done: merge augmentations (random cropping, color twitching)
        # TODO_Done: Do we need augmentation while eval here? Or adopt a different policy on cropping?
        label_quad, label_content, label_bool = load_annotation(label_path)
        valid_quad, valid_cont, valid_bool = check_and_validate_polys(label_quad, label_content,
                                                                      label_bool, img_ori_size)
        # Use float quads to generate maps
        if self.is_test_set:
            crop_img, crop_quad, crop_cont, crop_bool = random_crop(img, valid_quad, valid_cont, valid_bool,
                                                                    prob_bg=0.01, prob_partial=0.09)
            color_twitch_crop_img = Image.fromarray(np_img_color_twitch(np.array(crop_img),
                                                                        prob_1=0.4, prob_2=0.2, prob_3=0.10,
                                                                        max_twitch=0.25))
        else:
            crop_img, crop_quad, crop_cont, crop_bool = random_crop(img, valid_quad, valid_cont, valid_bool)
            color_twitch_crop_img = Image.fromarray(np_img_color_twitch(np.array(crop_img)))
        gen_img, resize_ratio = resize_image_fixed_square(color_twitch_crop_img)
        valid_quad_resized = np.array(crop_quad)
        if len(valid_quad_resized):
            valid_quad_resized[:, :, 0] *= resize_ratio[0]
            valid_quad_resized[:, :, 1] *= resize_ratio[1]
        score_map, geo_map, training_mask = generate_rbox(gen_img.size, valid_quad_resized,
                                                          valid_bool, valid_cont, img_path)
        # return img, valid_quad, valid_cont, score_map, geo_map, training_mask
        # Downsample the groundtruth to match the output size
        # print("Finishing img:", img_path)
        return self.toTensor(gen_img), torch.FloatTensor(score_map[::, ::4, ::4]), \
            torch.FloatTensor(geo_map[::, ::4, ::4]), torch.FloatTensor(training_mask[::, ::4, ::4])


class DataProvider:

    def __init__(self, batch_size=8, is_cuda=False, num_workers=config.data_loader_worker_num,
                 data_path=config.training_data_path_z440, with_text=False, test_set=False):
        self.batch_size = batch_size
        self.dataset = STV2KDetDataset(data_path, test_set, with_text)
        self.is_cuda = is_cuda
        self.data_iter = None
        self.iteration = 0  # iteration num of current epoch
        self.epoch = 0  # total epoch(s) finished
        self.num_workers = num_workers
        print("Image mean normalization is temporarily disabled.")
        # if "train" in data_path:
        #    self.image_channel_means = config.STV2K_train_image_channel_means
        # elif "test" in data_path:
        #     self.image_channel_means = config.STV2K_test_image_channel_means
        # else:
        #     self.image_channel_means = None
        # print("Image channel means set to ", self.image_channel_means, " for dataset on ", data_path)

    def build(self):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=self.num_workers, drop_last=True)
        self.data_iter = DataLoaderIter(data_loader)

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            # Disable normalization
            # if self.image_channel_means is not None:
            #     batch[0] = image_normalize(batch[0], self.image_channel_means)
            if self.is_cuda:
                for i in range(len(batch)):
                    if isinstance(batch[i], torch.Tensor):
                        batch[i] = batch[i].cuda()
            return batch

        except StopIteration:  # Reload after one epoch
            self.epoch += 1
            self.build()
            self.iteration = 1  # reset and return the 1st batch

            batch = self.data_iter.next()
            if self.is_cuda:
                for i in range(len(batch)):
                    if isinstance(batch[i], torch.Tensor):
                        batch[i] = batch[i].cuda()
            return batch


# DEPRECATED
# def crop_area(im, polys, tags, crop_background=False, max_tries=50):
#     """
#     make random crop from the input image
#     """
# def shrink_poly(poly, r):
#     """
#     fit a poly inside the origin poly, maybe bugs here...
#     used for generate the score map
#     """
# def generator(input_size=512, batch_size=4, background_ratio=3./8,
#               random_scale=np.array([0.5, 1, 2.0, 3.0]), vis=False):
# def get_batch(num_workers, **kwargs):
# class GeneratorEnqueuer:
#     """
#     Adopted from Keras Impl.
#     Builds a queue out of a data generator.
#     Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
#
#     # Arguments
#         generator: a generator function which endlessly yields data
#         use_multiprocessing: use multiprocessing if True, otherwise threading
#         wait_time: time to sleep in-between calls to `put()`
#         random_seed: Initial seed for workers,
#             will be incremented by one for each workers.
#     """
#


if __name__ == '__main__':
    # img = Image.open(config.__sample_path_1__)
    # img, o = resize_image(img, 3200)
    # print(img.size)
    # img.show()
    dp = DataProvider(3, num_workers=2, data_path="../testcases")
    while True:
        try:
            dp.next()
        except StopIteration:
            break
        except Exception as _Eall:
            print(_Eall)
        if dp.epoch > 0:
            break
