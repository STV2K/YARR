import os
import numpy as np
import tensorflow as tf
import config_utils as config
import data_utils as data_utils

from nets import STVNet

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = config.FLAGS.gpu_list


def main():
    data = get_tf_data(config.FLAGS.training_data_path, '/media/data2/hcx_data/STV2KTF/STV2K_0000.tfrecord')
    provider = slim.dataset_data_provider.DatasetDataProvider(
                    data,
                    num_readers=4,
                    common_queue_capacity=20 * 32,
                    common_queue_min=10 * 32,
                    shuffle=True)
    [image, x1, y1, x2, y2, x3, y3, x4, y4] = provider.get(['image', 
                                                            'object/x1', 'object/y1',
                                                            'object/x2', 'object/y2',
                                                            'object/x3', 'object/y3',
                                                            'object/x4', 'object/y4'])


    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')
    logits, end_points = STVNet.model(inputs)

    init = tf.global_variables_initializer()
    print(gpus)
    with tf.Session() as sess:
        sess.run(init)
        f_score, f_geo = sess.run([logits, end_points], feed_dict={inputs: [image]})