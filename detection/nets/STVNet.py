import tensorflow as tf
from collections import namedtuple
from . import resnet_v1
from . import resnet_utils
from . import custom_layers

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
      num_classes=1,
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

def model(images,
          num_classes=default_params.num_classes,
          feat_layers=default_params.feat_layers,
          anchor_sizes=default_params.anchor_sizes,
          anchor_ratios=default_params.anchor_ratios,
          normalizations=default_params.normalizations,
          prediction_fn=slim.softmax,
          reuse=reuse,
          weight_decay=1e-5,
          dropout_keep_prob=0.5,
          is_training=True):
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
            net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
            # Block 4: 1x1 conv. Because the fuck.
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            end_points['block5'] = net
            net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

            # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
            end_point = 'block6'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                net = custom_layers.pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block7'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = custom_layers.pad2d(net, pad=(1, 1))
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

            predictions = []
            logits = []
            localisations = []
            for i, layer in enumerate(feat_layers):
                with tf.variable_scope(layer + '_box'):
                    p, l = ssd_multibox_layer(end_points[layer],
                                              num_classes,
                                              anchor_sizes[i],
                                              anchor_ratios[i],
                                              normalizations[i])
                predictions.append(prediction_fn(p))
                logits.append(p)
                localisations.append(l)

        return predictions, localisations, logits, end_points

def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred


def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]