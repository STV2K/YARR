import os
import numpy as np
import tensorflow as tf
import config_utils as config

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = config.FLAGS.gpu_list