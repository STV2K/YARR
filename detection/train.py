import os
import numpy as np
import tensorflow as tf
import config_utils as config
import data_utils as data_utils

from nets import STVNet

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # config.FLAGS.gpu_list


def main():

    # im = tf.reshape(image, [3264, 2448, 3]) 
    print(height)
    image.set_shape([height, width, 3]) 
    b_image = tf.train.shuffle_batch([image], batch_size=4, capacity=20, min_after_dequeue=10)
    
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')
    logits, end_points = STVNet.model(inputs)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(image)
        threads = tf.train.start_queue_runners(sess=sess)

        print(b_image)
        b_image = sess.run(b_image)
        f_score, f_geo = sess.run([logits, end_points], feed_dict={inputs: b_image})

if __name__ == '__main__':
    main()
