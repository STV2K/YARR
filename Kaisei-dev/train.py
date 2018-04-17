#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import models
import torchvision
import config

gpus = list(range(len(config.gpu_list.split(','))))


def train():
    # TODO: checkpoint, gpu_config, restoring
    input_images = []
    input_score_maps = []
    input_geo_maps = []
    input_training_masks = []  # What's this?



