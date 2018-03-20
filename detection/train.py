import os
import numpy as np
import tensorflow as tf
import config_utils as config
import data_utils as data_utils

from nets import STVNet

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # config.FLAGS.gpu_list


def main():

    image, x1, bbox_num = data_utils.read_and_decode()
    
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')
    logits, end_points = STVNet.model(inputs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        b_image, b_x1, b_bbox_num = sess.run([image, x1, bbox_num])
        f_score, f_geo = sess.run([logits, end_points], feed_dict={inputs: b_image[0]})

        print(f_geo)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
