import tensorflow as tf
from collections import namedtuple
from . import resnet_v1
from . import resnet_utils

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
      feat_layers=['block2', 'block7', 'block8', 'block9', 'block10', 'block11'],
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

def model(images, weight_decay=1e-5, is_training=True):
  with slim.arg_scope(resnet_utils.resnet_arg_scope(weight_decay=weight_decay)):
        net, end_points = resnet_v1.resnet_v1_50(images,
                              is_training=is_training,
                              scope='resnet_v1_50')
        print('block 1 shape: %s' % end_points['resnet_v1_50/block1'].shape)
        print('block 2 shape: %s' % end_points['resnet_v1_50/block2'].shape)
        print('block 3 shape: %s' % end_points['resnet_v1_50/block3'].shape)
        print('block 4 shape: %s' % end_points['resnet_v1_50/block4'].shape)
        # TODO: add layer
        # net = end_points['resnet_v1_50/block2']
        # net = slim.conv2d(net, 512, [3, 3], padding=1)

        return net, end_points
