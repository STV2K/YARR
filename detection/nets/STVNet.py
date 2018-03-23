import tensorflow as tf
from collections import namedtuple
from . import resnet_v1
from . import resnet_utils

from tensorflow.contrib.framework.python.ops import add_arg_scope


slim = tf.contrib.slim

STVParams = namedtuple('STVParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])

default_params = STVParams(
      img_shape=(300, 300),
      num_classes=21,
      no_annotation_label=21,
      feat_layers=['resnet_v1_50/block2', 'block5', 'block6', 'block7', 'block8', 'block9'],
      feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
      anchor_size_bounds=[0.15, 0.90],
      # anchor_size_bounds=[0.20, 0.90],
      anchor_sizes=[(21., 45.),
                    (45., 99.),
                    (99., 153.),
                    (153., 207.),
                    (207., 261.),
                    (261., 315.)],
      anchor_ratios=[[2, .5],
                     [2, .5, 3, 1./3],
                     [2, .5, 3, 1./3],
                     [2, .5, 3, 1./3],
                     [2, .5],
                     [2, .5]],
      anchor_steps=[8, 16, 32, 64, 100, 300],
      anchor_offset=0.5,
      normalizations=[20, -1, -1, -1, -1, -1],
      prior_scaling=[0.1, 0.1, 0.2, 0.2]
      )

@add_arg_scope
def pad2d(inputs,
          pad=(0, 0),
          mode='CONSTANT',
          data_format='NHWC',
          trainable=True,
          scope=None):
    """2D Padding layer, adding a symmetric padding to H and W dimensions.

    Aims to mimic padding in Caffe and MXNet, helping the port of models to
    TensorFlow. Tries to follow the naming convention of `tf.contrib.layers`.

    Args:
      inputs: 4D input Tensor;
      pad: 2-Tuple with padding values for H and W dimensions;
      mode: Padding mode. C.f. `tf.pad`
      data_format:  NHWC or NCHW data format.
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        # Padding shape.
        if data_format == 'NHWC':
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net

def stv_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:
            return sc

def model(images, weight_decay=1e-5, is_training=True):
  with slim.arg_scope(resnet_utils.resnet_arg_scope(weight_decay=weight_decay)):
        net, end_points = resnet_v1.resnet_v1_50(images,
                              is_training=is_training,
                              scope='resnet_v1_50')

        # TODO: add layer
        arg_scope = stv_arg_scope(weight_decay=0.00004)
        with slim.arg_scope(arg_scope):
            # Block 3: let's dilate the hell out of it!
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
            end_points['block4'] = net
            net = tf.layers.dropout(net, rate=0.5, training=is_training)
            # Block 4: 1x1 conv. Because the fuck.
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            end_points['block5'] = net
            net = tf.layers.dropout(net, rate=0.5, training=is_training)

            # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
            end_point = 'block6'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                net = pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block7'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block9'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net

        # net = end_points['resnet_v1_50/block2']
        # net = slim.conv2d(net, 512, [3, 3], padding=1)

        return net, end_points
