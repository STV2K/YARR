from nets import resnet_v1
from nets import resnet_utils
from PIL import Image
import numpy as np
import tensorflow as tf
import config_utils as config

slim = tf.contrib.slim
STV2K_Path = '/media/data2/hcx_data/STV2K/stv2k_train/'
default_size = resnet_v1.resnet_v1.default_image_size
gpu_list = config.FLAGS.gpu_list.split(',')
gpus = [int(gpu_list[i]) for i in range(len(gpu_list))]

def get_image(img_path):
    im = Image.open(img_path)
    im = im.resize(default_size)
    im = np.array(im)
    # img_input = tf.to_float(tf.convert_to_tensor(im))

    return im

def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2, tf.shape(inputs)[2]*2])

def run_once(inputs, weight_decay=1e-5, is_training=True):
    with slim.arg_scope(resnet_utils.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(inputs, is_training=True, scope = 'resnet_v1_50')
    return logits, end_points

def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    # images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_utils.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * config.FLAGS.text_scale
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    print('F_score : ', F_score)
    print('F_geometry : ', F_geometry)
    return F_score, F_geometry


if __name__ == "__main__":
    img_name = 'STV2K_tr_0001.jpg'
    im = get_image(STV2K_Path + img_name)

    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='inputs')
    # logits, end_points = run_once(inputs)
    logits, end_points = model(inputs)

    init = tf.global_variables_initializer()
    # with(tf.device('/gpu:%d' % gpus[0])):
    print(gpus)
    with tf.Session() as sess:
        sess.run(init)
        # logs, ends = sess.run([logits, end_points], feed_dict={inputs: [im]})
        f_score, f_geo = sess.run([logits, end_points], feed_dict={inputs: [im]})
    
    # block1 = ends['resnet_v1_50/block4']
    # map1 = block1[0,:,:,0]
    # print(map1)
    

