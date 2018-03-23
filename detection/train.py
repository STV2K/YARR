import os
import numpy as np
import tensorflow as tf
import config_utils as config
import data_utils as data_utils

from nets import STVNet

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # config.FLAGS.gpu_list


def main():

    image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num = data_utils.read_and_decode()
    
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')
    logits, end_points = STVNet.model(inputs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        b_image, b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num = \
            sess.run([image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num])
        f_score, f_geo = sess.run([logits, end_points], feed_dict={inputs: [b_image[0]]})

        print('block 1 shape: ',  f_geo['resnet_v1_50/block1'].shape)
        print('block 2 shape: ',  f_geo['resnet_v1_50/block2'].shape)
        print('block 3 shape: ',  f_geo['block3'].shape)
        print('block 4 shape: ',  f_geo['block4'].shape)
        print('block 5 shape: ',  f_geo['block5'].shape)
        print('block 6 shape: ',  f_geo['block6'].shape)
        print('block 7 shape: ',  f_geo['block7'].shape)
        print('block 8 shape: ',  f_geo['block8'].shape)
        print(f_score.shape, b_bbox_num[0])

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
