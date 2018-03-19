import os
import numpy as np
import tensorflow as tf
import config_utils as config
import data_utils as data_utils

from nets import STVNet

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # config.FLAGS.gpu_list


def main():
    data = data_utils.get_tf_data(config.FLAGS.training_data_path, '/media/data2/hcx_data/STV2KTF/STV2K_0006.tfrecord')
    provider = slim.dataset_data_provider.DatasetDataProvider(
                    data,
                    num_readers=4,
                    common_queue_capacity=20 * 32,
                    common_queue_min=10 * 32,
                    shuffle=True)
    [image, height, width, x1, y1, x2, y2, x3, y3, x4, y4] = provider.get([
                                                            'image', 'height', 'width', 
                                                            'object/x1', 'object/y1',
                                                            'object/x2', 'object/y2',
                                                            'object/x3', 'object/y3',
                                                            'object/x4', 'object/y4'])


    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')
    logits, end_points = STVNet.model(inputs)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(image)
        # im = tf.reshape(image, [3264, 2448, 3]) 
        # inputimg = sess.run(im)
        r = tf.train.batch(reshape_list([image]), batch_size=4, num_threads=4, capacity=20)
        b_image = reshape_list(r, [3264, 2448, 3])
        print(b_image)
        f_score, f_geo = sess.run([logits, end_points], feed_dict={inputs: b_image})

def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.

    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r

if __name__ == '__main__':
    main()
