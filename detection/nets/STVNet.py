import math
import tensorflow as tf
import numpy as np
import tf_extended as tfe
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
                                         'prior_scaling',
                                         'mbox_kernel'
                                         ])

default_params = STVParams(
      img_shape=(300, 300),
      num_classes=2,
      no_annotation_label=21,
      feat_layers=['stvnet/resnet_v1_50/block1', 'block5', 'block6', 'block7', 'block8', 'block9'],
      feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
      anchor_size_bounds=[0.15, 0.90],
      # anchor_size_bounds=[0.20, 0.90],
      # anchor_sizes=[(21., 45.),
      #               (45., 99.),
      #               (99., 153.),
      #               (153., 207.),
      #               (207., 261.),
      #               (261., 315.)],
      anchor_sizes=[(0.07, 0.15),
                    (0.15, 0.33),
                    (0.33, 0.51),
                    (0.51, 0.69),
                    (0.69, 0.87),
                    (0.87, 1.05)],
      anchor_ratios=[[2, .5, 5],
                     [2, .5, 3, 1./3, 5],
                     [2, .5, 3, 1./3, 5],
                     [2, .5, 3, 1./3, 5],
                     [2, .5, 5],
                     [2, .5, 5]],     # add long ratio boxes
      anchor_steps=[8, 16, 32, 64, 128, 256],
      anchor_offset=0.5,
      normalizations=[20, -1, -1, -1, -1, -1],
      prior_scaling=[0.1, 0.1, 0.2, 0.2],
      mbox_kernel = [(3, 5),
                     (3, 5),
                     (3, 5),
                     (3, 5),
                     (3, 5),
                     (1, 1)]
      )

def redefine_params(img_width, img_height):
    default_bk = default_params
    default_params = STVParams(
        img_shape=(img_height, img_width),
        num_classes=default_bk.num_classes,
        feat_layers=default_bk.feat_layers,
        feat_shape=[(math.ceil(img_height / 8), math.ceil(img_width / 8)),
                    (math.ceil(img_height / 16), math.ceil(img_width / 16)),
                    (math.ceil(img_height / 32), math.ceil(img_width / 32)),
                    (math.ceil(img_height / 64), math.ceil(img_width / 64)),
                    (math.ceil(img_height / 128), math.ceil(img_width / 128)),
                    (math.ceil(img_height / 256), math.ceil(img_width / 256))],
        anchor_size_bounds=default_bk.anchor_size_bounds,
        anchor_sizes=default_bk.anchor_sizes,
        anchor_ratios=default_bk.anchor_ratios,
        anchor_steps=default_bk.anchor_steps,
        anchor_offset=default_bk.anchor_offset,
        normalizations=default_bk.normalizations,
        prior_scaling=default_bk.prior_scaling
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
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc

def model(images,
          num_classes=default_params.num_classes,
          feat_layers=default_params.feat_layers,
          anchor_sizes=default_params.anchor_sizes,
          anchor_ratios=default_params.anchor_ratios,
          normalizations=default_params.normalizations,
          kernels=default_params.mbox_kernel,
          prediction_fn=slim.softmax,
          reuse=None,
          weight_decay=1e-5,
          dropout_keep_prob=0.5,
          is_training=True):

    arg_scope = stv_arg_scope(weight_decay=0.00004)
    with slim.arg_scope(arg_scope):

    #with slim.arg_scope(resnet_utils.resnet_arg_scope(weight_decay=weight_decay)):
        
        with tf.variable_scope('stvnet', 'stvnet', [images], reuse=reuse):
            net, end_points = resnet_v1.resnet_v1_50(images,
                                  is_training=is_training,
                                  scope='resnet_v1_50')
            
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
                                              normalizations[i],
                                              kernels[i])
                predictions.append(prediction_fn(p))
                logits.append(p)
                localisations.append(l)

        return predictions, localisations, logits, end_points

# for each feature map, calculate the prediction
def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       kernel=[1, 1],
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
    loc_pred = slim.conv2d(net, num_loc_pred, kernel, activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, kernel, activation_fn=None,
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


# return anchors
def ssd_anchors_all_layers(img_shape=default_params.img_shape,
                           layers_shape=default_params.feat_shapes,
                           anchor_sizes=default_params.anchor_sizes,
                           anchor_ratios=default_params.anchor_ratios,
                           anchor_steps=default_params.anchor_steps,
                           offset=default_params.anchor_offset,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# return anchor of one layer
def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    # h[0] = sizes[0] / img_shape[0]
    # w[0] = sizes[0] / img_shape[1]
    h[0] = sizes[0]
    w[0] = sizes[0]
    di = 1
    if len(sizes) > 1:
        # h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        # w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        h[1] = math.sqrt(sizes[0] * sizes[1])
        w[1] = math.sqrt(sizes[0] * sizes[1])
        di += 1
    for i, r in enumerate(ratios):
        # h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        # w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
        h[i+di] = sizes[0] / math.sqrt(r)
        w[i+di] = sizes[0] * math.sqrt(r)
    return y, x, h, w


def detected_bboxes(predictions, localisations,
                    select_threshold=None, nms_threshold=0.5,
                    clipping_bbox=None, top_k=400, keep_top_k=200):
    """Get the detected bounding boxes from the SSD network output.
    """
    # Select top_k bboxes from predictions, and clip
    rscores, rbboxes = \
        tf_ssd_bboxes_select(predictions, localisations,
                             select_threshold=select_threshold,
                             num_classes=default_params.num_classes)
    rscores, rbboxes = \
        tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
    # Apply NMS algorithm.
    rscores, rbboxes = \
        tfe.bboxes_nms_batch(rscores, rbboxes,
                             nms_threshold=nms_threshold,
                             keep_top_k=keep_top_k)
    if clipping_bbox is not None:
        rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
    return rscores, rbboxes


def tf_ssd_bboxes_select_layer(predictions_layer, localizations_layer,
                               select_threshold=None,
                               num_classes=21,
                               ignore_class=0,
                               scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tfe.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tfe.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def tf_ssd_bboxes_select(predictions_net, localizations_net,
                         select_threshold=None,
                         num_classes=21,
                         ignore_class=0,
                         scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i],
                                                        localizations_net[i],
                                                        select_threshold,
                                                        num_classes,
                                                        ignore_class)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes


def tf_ssd_bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes


def tf_ssd_bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=default_params.prior_scaling,
                         scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_ssd_bboxes_decode_layer(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes


# encode gt
def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes=default_params.num_classes,
                         # no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores = \
                    tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               num_classes, # no_annotation_label,
                                               ignore_threshold,
                                               prior_scaling, dtype)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
        return target_labels, target_localizations, target_scores

def tf_ssd_bboxes_batch_encode(labels,
                         bboxes,
                         anchors,
                         batch_size,
                         num_classes=default_params.num_classes,
                         # no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      labels: BatchSizex1D Tensor(int64) containing groundtruth labels;
      bboxes: BatchSizexNx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                layer_labels = []
                layer_localizations = []
                layer_scores = []
                for j in range(batch_size):
                    t_labels, t_loc, t_scores = \
                        tf_ssd_bboxes_encode_layer(labels[j], bboxes[j], anchors_layer,
                                                   num_classes, # no_annotation_label,
                                                   ignore_threshold,
                                                   prior_scaling, dtype)
                    layer_labels.append(t_labels)
                    layer_localizations.append(t_loc)
                    layer_scores.append(t_scores)

                layer_labels = tf.stack(layer_labels)
                layer_localizations = tf.stack(layer_localizations)
                layer_scores = tf.stack(layer_scores)
                target_labels.append(layer_labels)
                target_localizations.append(layer_localizations)
                target_scores.append(layer_scores)
        return target_labels, target_localizations, target_scores

# encode gt for one layer
def tf_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               # no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        """Compute intersection between score a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.div(inter_vol, vol_anchors)
        return scores

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        # Check no annotation label: ignore these anchors...
        # interscts = intersection_with_anchors(bbox)
        # mask = tf.logical_and(interscts > ignore_threshold,
        #                       label == no_annotation_label)
        # # Replace scores by -1.
        # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores


def ssd_losses(logits, localisations,
           gclasses, glocalisations, gscores,
           match_threshold=0.5,
           negative_ratio=3.,
           alpha=1.,
           label_smoothing=0.,
           device='/cpu:0',
           scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tfe.get_shape(tf.convert_to_tensor(logits[0]), 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        #print('logits: ', logits)
        #print('localisations: ', localisations)
        #print('gclasses: ', gclasses)
        #print('glocalisations: ', glocalisations)
        #print('gscores: ', gscores)

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gscores > match_threshold    # gt boxes > threshold     true or false
        fpmask = tf.cast(pmask, dtype)       # float version of positive gt boxes
        n_positives = tf.reduce_sum(fpmask)  # number of positive gt boxes

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)         # int version of positive gt boxes
        predictions = slim.softmax(logits)            # softmax score predictions
        nmask = tf.logical_and(tf.logical_not(pmask), # negtive mask
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)                # float version of negtive mask
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)    # number of negtives gt boxes
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)    # find the suitable number of negtive gt boxes

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)   # find the negtive boxes which have the highest score, val < 0
        max_hard_pred = -val[-1]    # find the one which has the largest score in negtive boxes
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)  # find the negtive boxes we need(follow the ratio)
        fnmask = tf.cast(nmask, dtype)

        batch_size = tf.cast(batch_size, tf.float32)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)
            #ret_loss1 = loss
            loss = tf.div(tf.reduce_sum(loss * fpmask), n_positives, name='value') #batch_size, name='value')
            #ret_loss2 = loss
            tf.losses.add_loss(loss)

            pos_loss = loss

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss = tf.div(tf.reduce_sum(loss * fnmask), n_positives, name='value') #batch_size, name='value')
            tf.losses.add_loss(loss)

            neg_loss = loss

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            loss = tf.div(tf.reduce_sum(loss * weights), n_positives, name='value') #batch_size, name='value')
            tf.losses.add_loss(loss)

            loc_loss = loss
        
        # add return to run
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regular_loss = tf.add_n(regularization_loss)
        return pos_loss, neg_loss, loc_loss, regular_loss #, ret_loss1, ret_loss2
