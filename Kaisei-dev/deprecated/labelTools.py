#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import os
import numpy as np
from shapely.geometry import Polygon

class LabelReader:

    @staticmethod
    def read_stv2k_label(label_path):
        """
        Load the new STV2k label file into arrays.
        @return *x_axis*, *y_axis* and *content*.

        The label file ought to be encoded with GB2312 and organized as:
            [Line 1] x1, y1, x2, y2, x3, y3, x4, y4 (all integers with origin on the top left)
                   (TopRight)(BtmR)  (BtmL) (TopLeft)
            [Line 2] Text contents or blank line if unrecognizable
            [Line 3] Blank line
            [Repeat for other bounding boxes]

        NOTICE: Some indices in the label files may be out of boundary, which are NOT tackled here.
        """
        x_axis = []
        y_axis = []
        content = []
        with open(label_path, "r", encoding="GB2312") as lf:
            count = 0
            for line in lf:
                if count % 3 == 0 and line != "":
                    line = line.strip().split(",")
                    x_axis.append(list(map(int, line[0:-1:2])))
                    y_axis.append(list(map(int, line[1::2])))
                if count % 3 == 1:
                    content.append(line.strip())
                count += 1
        return np.array(x_axis).astype(int), np.array(y_axis).astype(int), content


# class labelTransformer:
#     def transSTV2ktoFOTSdet(self, xAxis, yAxis, content, width = ):
#         '''
#         Transform STV2k labels to FOTS style notations:
#         1/4 sized pixel-wise six-channel output,
#         describing probability, relative distance to top/bottom/left/right and orientation.
#         @return *outmap*, a float array of size 6 * (w/4) * (h/4)
#                 **
#
#         @policy + Is the distance relative or absolute? = Using relative dist
#                 + Don't Care channel                    = Removing all blank-content labels
#                 + Overlapping labels                    = Prob to 1, max(btmAndRightDist), min(topAndLeft),
#                                                           set orientation as nearest label center
#                 + Distinguishing vertical texts and
#                   horizontal texts with 90-angle orientation  = Need further discussions
#                   -- do we tackle this in detection (mark as diff type)
#                      or recognition (robust to mirrored/vertical/horizontal text)?
#         '''
#

# TODO: quad2rbox