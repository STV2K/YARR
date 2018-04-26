import os
from PIL import Image
import sys
import cv2
import random
import numpy as np
import tensorflow as tf
import tf_extended as tfe
import config_utils as config

from shapely.geometry import Polygon
from tensorflow.python.ops import array_ops

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

img_width = config.FLAGS.input_size_width
img_height = config.FLAGS.input_size_height
train_image_path = '/media/data1/hcxiao/ICDAR15-IncidentalSceneText/ch4_train_images/'
train_gt_path = '/media/data1/hcxiao/ICDAR15-IncidentalSceneText/ch4_train_gts/'


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]

def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2)
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle

def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

def get_parallelogram(poly):
    fitted_parallelograms = []
    for i in range(4):
        p0 = poly[i]
        p1 = poly[(i + 1) % 4]
        p2 = poly[(i + 2) % 4]
        p3 = poly[(i + 3) % 4]
        edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
        backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
        forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
        if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
            # 平行线经过p2
            if edge[1] == 0:
                edge_opposite = [1, 0, -p2[0]]
            else:
                edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
        else:
            # 经过p3
            if edge[1] == 0:
                edge_opposite = [1, 0, -p3[0]]
            else:
                edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
        # move forward edge
        new_p0 = p0
        new_p1 = p1
        new_p2 = p2
        new_p3 = p3
        new_p2 = line_cross_point(forward_edge, edge_opposite)
        if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
            # across p0
            if forward_edge[1] == 0:
                forward_opposite = [1, 0, -p0[0]]
            else:
                forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
        else:
            # across p3
            if forward_edge[1] == 0:
                forward_opposite = [1, 0, -p3[0]]
            else:
                forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
        new_p0 = line_cross_point(forward_opposite, edge)
        new_p3 = line_cross_point(forward_opposite, edge_opposite)
        fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        # or move backward edge
        new_p0 = p0
        new_p1 = p1
        new_p2 = p2
        new_p3 = p3
        new_p3 = line_cross_point(backward_edge, edge_opposite)
        if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
            # across p1
            if backward_edge[1] == 0:
                backward_opposite = [1, 0, -p1[0]]
            else:
                backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
        else:
            # across p2
            if backward_edge[1] == 0:
                backward_opposite = [1, 0, -p2[0]]
            else:
                backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
        new_p1 = line_cross_point(backward_opposite, edge)
        new_p2 = line_cross_point(backward_opposite, edge_opposite)
        fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
    areas = [Polygon(t).area for t in fitted_parallelograms]
    parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
    return parallelogram

def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)
            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)
            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)
            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)
            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)
            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)
            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)
            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)
            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)

def sort_poly(poly):
    edge = [
            (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
            ]
    area =  np.sum(edge) / 2.

    if area > 0:
        poly = poly
    else:
        poly = poly[(0, 3, 2, 1), :]

    return poly

def get_angle(poly):
    poly = np.array(poly)
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        return 0.
    else:
        p_lowest_right = (p_lowest - 1) % 4
        divide_num = (poly[p_lowest][0] - poly[p_lowest_right][0])
        if divide_num == 0:
            angle = 0
        else:
            angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1]) / divide_num)

        if angle / np.pi * 180 > 45:
            return -(np.pi / 2 - angle)
        else:
            return angle


def get_poly_angle(poly):
    poly = np.array(poly)
    poly = get_parallelogram(poly)
    poly = sort_poly(poly)
    poly = rectangle_from_parallelogram(poly)
    angle = get_angle(poly)
    return angle


def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    """Resize an image and bounding boxes.
    """
    # Resize image.
    with tf.name_scope('resize_image'):
        height, width, channels = _ImageDimensions(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image

def _ImageDimensions(image):
    """Returns the dimensions of an image tensor.
    Args:
      image: A 3-D Tensor of shape `[height, width, channels]`.
    Returns:
      A list of `[height, width, channels]` corresponding to the dimensions of the
        input image.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = tfe.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = tfe.bboxes_filter_overlap(labels, bboxes,
                                                   threshold=0.5,#BBOX_CROP_OVERLAP,
                                                   assign_negative=False)
        return cropped_image, labels, bboxes, distort_bbox

def get_processed_imgs(images,
                       labels,
                       bboxes):
    ret_images = []
    ret_labels = []
    ret_bboxes = []
    for i in range(3):
        cropped_image, cropped_labels, cropped_bboxes, distort_box = distorted_bounding_box_crop(images[i], labels[i], bboxes[i],
                                                                                    min_object_covered=0.25,
                                                                                    aspect_ratio_range=(0.6, 1.67))
        cropped_image = resize_image(cropped_image, (300,300),
                                     method=tf.image.ResizeMethod.BILINEAR,
                                     align_corners=False)

        ret_images.append(cropped_image)
        ret_labels.append(cropped_labels)
        ret_bboxes.append(cropped_bboxes)
        
    ret_images = tf.stack(ret_images)
    ret_labels = tf.stack(ret_labels)
    ret_bboxes = tf.stack(ret_bboxes)
#    ret_images = np.array(ret_images)
#    ret_labels = np.array(ret_labels)
#    ret_bboxes = np.array(ret_bboxes)
    return ret_images, ret_labels, ret_bboxes


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))



def get_images():
    files = []
    for root, dirs, filenames in os.walk(train_image_path):
        for file in filenames:
            if not file.endswith('.jpg'):
                continue
            files.append(os.path.join(root, file))
    print('{} training images in {}'.format(len(files), train_image_path))
    return files

def load_annotation_STV2K(ano_path):
    polys = []
    tags = []
    objIndex = 0
    datas = ""
    gt_word = ""
    with open(ano_path, 'r', encoding="gbk") as file:
        for line in file.readlines():
            objIndex += 1

            line_num = objIndex % 3
            if line_num == 1:
                data = line
                continue
            elif line_num == 2:
                gt_word = line.strip('\r\n')
            else:
                continue

            datas = data.split(',')
            nums = []
            for num in datas:
                nums.append(int(num))
            x1, y1, x2, y2, x3, y3, x4, y4 = nums
            polys.append((x1, y1, x2, y2, x3, y3, x4, y4))

            if gt_word != "" and gt_word != "\n" and gt_word != "\r\n":
                tags.append(True)
            else:
                tags.append(False)

    return polys, tags

def load_annotation(ano_path):
    polys = []
    tags = []
    with open(ano_path, 'r', encoding="utf-8-sig") as file:
        for line in file.readlines():
            line = line.strip('\r\n')
            line = line.strip('\n')
            line = line.strip(',')  # there is a ',' at the end of the line in some annotation files
            datas = line.split(',')

            if datas[-1] == '###':
                tags.append(False)
#                continue
            else:
                tags.append(True)

            nums = []
            #print(datas)
            for data in datas[0:8]:
                nums.append(int(data))
            x1, y1, x2, y2, x3, y3, x4, y4 = nums
            polys.append((x1, y1, x2, y2, x3, y3, x4, y4))
    return polys, tags


def process_image(filename):
    # image_data = tf.gfile.FastGFile(filename, 'rb').read()
    img = np.array(Image.open(filename))
    image_data = img.tostring()
    im = cv2.imread(filename)
    shape = im.shape

    #anno_filename = filename.replace('.jpg', '.txt')
    anno_filename = train_gt_path + 'gt_' + filename.split('/')[-1].replace('jpg', 'txt')
    bboxes, tags = load_annotation(anno_filename)

    return image_data, shape, bboxes, tags


def convert_to_example(image_data, shape, bboxes, difficults):

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    x4 = []
    y4 = []
    for b in bboxes:
        [l.append(point) for l, point in zip([x1, y1, x2, y2, x3, y3, x4, y4], b)]

    example = tf.train.Example(features=tf.train.Features(feature={
              'image/height': int64_feature(shape[0]),
              'image/width': int64_feature(shape[1]),
              'image/channels': int64_feature(shape[2]),
              'image/object/bbox/num': int64_feature(len(x1)),
              # 'image/shape': int64_feature(shape),
              'image/object/bbox/x1': float_feature(x1),
              'image/object/bbox/y1': float_feature(y1),
              'image/object/bbox/x2': float_feature(x2),
              'image/object/bbox/y2': float_feature(y2),
              'image/object/bbox/x3': float_feature(x3),
              'image/object/bbox/y3': float_feature(y3),
              'image/object/bbox/x4': float_feature(x4),
              'image/object/bbox/y4': float_feature(y4),
              # 'image/object/difficult': 
              'image/encoded': bytes_feature(image_data),
              'image/format': bytes_feature(b'jpg')
              }))
    return example


def add_to_tfrecord(filename, tfrecord_writer):
    image_data, shape, bboxes, difficults = process_image(filename)
    example = convert_to_example(image_data, shape, bboxes, difficults)
    tfrecord_writer.write(example.SerializeToString())


SAMPLE_PER_FILE = 100

def run_STV2K(output_dir, shuffling=False, name='STV2K'):
    filenames = get_images()

    if shuffling:
        random.seed()
        random.shuffle(filenames)

    i = 0
    index = 0
    files_len = len(filenames)
    while i < files_len:
        tf_filename = "%s%s_%04d.tfrecord" % (output_dir, name, index)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < files_len and j < SAMPLE_PER_FILE:
                sys.stdout.write("\r>> Converting image %d/%d" % (i+1, files_len))
                sys.stdout.flush()

                filename = filenames[i]
                add_to_tfrecord(filename, tfrecord_writer)
                i += 1
                j += 1
            index += 1

    print('\nFinish converting datasets')

def run(output_dir, shuffling=False, name='icdar'):
    filenames = get_images()

    if shuffling:
        random.seed()
        random.shuffle(filenames)

    i = 0
    index = 0
    files_len = len(filenames)
    while i < files_len:
        tf_filename = "%s%s_%04d.tfrecord" % (output_dir, name, index)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < files_len and j < SAMPLE_PER_FILE:
                sys.stdout.write("\r>> Converting image %d/%d" % (i+1, files_len))
                sys.stdout.flush()

                filename = filenames[i]
#                sys.stdout.write('\r\n' + filename)
                add_to_tfrecord(filename, tfrecord_writer)
                i += 1
                j += 1
            index += 1

    print('\nFinish converting datasets')


# IMAGE_HEIGHT = 300
# IMAGE_WIDTH = 300

# filenames = '/media/data2/hcx_data/STV2KTF/STV2K_0000.tfrecord'
stv2k_filenames = ['/media/data1/hcxiao/TFRecorders/STV2KTF/STV2K_0000.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/STV2KTF/STV2K_0001.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/STV2KTF/STV2K_0002.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/STV2KTF/STV2K_0003.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/STV2KTF/STV2K_0004.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/STV2KTF/STV2K_0005.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/STV2KTF/STV2K_0006.tfrecord']
icdar_filenames = ['/media/data1/hcxiao/TFRecorders/ICDAR15TF/icdar_0000.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/ICDAR15TF/icdar_0001.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/ICDAR15TF/icdar_0002.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/ICDAR15TF/icdar_0003.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/ICDAR15TF/icdar_0004.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/ICDAR15TF/icdar_0005.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/ICDAR15TF/icdar_0006.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/ICDAR15TF/icdar_0007.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/ICDAR15TF/icdar_0008.tfrecord',
                   '/media/data1/hcxiao/TFRecorders/ICDAR15TF/icdar_0009.tfrecord']
train_filenames = stv2k_filenames
val_filenames = ['/media/data1/hcxiao/TFRecorders/STV2KTF/STV2K_0003.tfrecord']


def read_data(train=True):
    if train:
        return read_and_decode(train_filenames)
    else:
        return read_and_decode(val_filenames)

def read_and_decode(filenames_string):
    filename_queue = tf.train.string_input_producer(filenames_string)
                                                     # num_epochs=10)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/object/bbox/num': tf.FixedLenFeature([1], tf.int64),
        # 'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/x1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y4': tf.VarLenFeature(dtype=tf.float32),
        # 'image/object/difficult': 
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg')})

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    bbox_num = tf.cast(features['image/object/bbox/num'], tf.int32)
    x1 = tf.sparse_tensor_to_dense(features['image/object/bbox/x1'])
    y1 = tf.sparse_tensor_to_dense(features['image/object/bbox/y1'])
    x2 = tf.sparse_tensor_to_dense(features['image/object/bbox/x2'])
    y2 = tf.sparse_tensor_to_dense(features['image/object/bbox/y2'])
    x3 = tf.sparse_tensor_to_dense(features['image/object/bbox/x3'])
    y3 = tf.sparse_tensor_to_dense(features['image/object/bbox/y3'])
    x4 = tf.sparse_tensor_to_dense(features['image/object/bbox/x4'])
    y4 = tf.sparse_tensor_to_dense(features['image/object/bbox/y4'])

    bbox_shape = tf.stack([bbox_num[0]])
    x1 = tf.reshape(x1, bbox_shape)
    y1 = tf.reshape(y1, bbox_shape)
    x2 = tf.reshape(x2, bbox_shape)
    y2 = tf.reshape(y2, bbox_shape)
    x3 = tf.reshape(x3, bbox_shape)
    y3 = tf.reshape(y3, bbox_shape)
    x4 = tf.reshape(x4, bbox_shape)
    y4 = tf.reshape(y4, bbox_shape)

    image_shape = tf.stack([height[0], width[0], 3])
    image = tf.reshape(image, image_shape)
    image_size_const = tf.constant((img_height, img_width, 3), dtype=tf.int32)
    # resize_image = tf.image.resize_image_with_crop_or_pad(image=image,
    #                                                       target_height=IMAGE_HEIGHT,
    #                                                       target_width=IMAGE_WIDTH)
    resize_image = tf.image.resize_images(image, size=[img_height, img_width])

    resize_ratio_x = tf.cast(img_width / width[0], tf.float32)
    resize_ratio_y = tf.cast(img_height / height[0], tf.float32)
    x1_r = x1 * resize_ratio_x
    x2_r = x2 * resize_ratio_x
    x3_r = x3 * resize_ratio_x
    x4_r = x4 * resize_ratio_x
    y1_r = y1 * resize_ratio_y
    y2_r = y2 * resize_ratio_y
    y3_r = y3 * resize_ratio_y
    y4_r = y4 * resize_ratio_y

    # image.set_shape([height[0], width[0], 3])
    images, x1_rs, x2_rs, x3_rs, x4_rs, y1_rs, y2_rs, y3_rs, y4_rs, bbox_nums = \
        tf.train.batch([resize_image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num],
                        batch_size=config.FLAGS.batch_size,
                        capacity=30,
                        num_threads=2,
                        # min_after_dequeue=10,
                        dynamic_pad=True)
    return images, x1_rs, x2_rs, x3_rs, x4_rs, y1_rs, y2_rs, y3_rs, y4_rs, bbox_nums


def test_get_image_annotation():
    get_images()
    polys = load_annotation('/home/hcxiao/Datasets/STV2k/stv2k_train/STV2K_tr_0001.txt')
    print(polys)


def generate_tfrecord():
    #run_STV2K("/media/data2/hcx_data/STV2KTF/", shuffling=True)
    run("/media/data2/hcx_data/ICDARTF/", shuffling=True)


def test_read():
    image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num = read_and_decode()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img, x1_rs, x2_rs, x3_rs, x4_rs, y1_rs, y2_rs, y3_rs, y4_rs, bbox_nums = \
            sess.run([image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num])
        print(x1_rs[0])
        print(x2_rs[0])
        print(x3_rs[0])
        print(x4_rs[0])
        print(y1_rs[0])
        print(y2_rs[0])
        print(y3_rs[0])
        print(y4_rs[0])

        coord.request_stop()
        coord.join(threads)

def turn_into_bbox(x1, x2, x3, x4, y1, y2, y3, y4, num):
    bboxes = []
    for i in range(num):
        x = [x1[i], x2[i], x3[i], x4[i]]
        y = [y1[i], y2[i], y3[i], y4[i]]
        xmin = min(x) / img_width
        xmax = max(x) / img_width
        ymin = min(y) / img_height
        ymax = max(y) / img_height

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > 1:
            xmax = 1
        if ymax > 1:
            ymax = 1

        bbox = [ymin, xmin, ymax, xmax]
        bboxes.append(bbox)

    return bboxes

def generate_batch_bboxes(b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num):
    batch_bboxes = []
    batch_labels = []
    max_num = 0
    for i in range(len(b_bbox_num)):
        if max_num < b_bbox_num[i][0]:
            max_num = b_bbox_num[i][0]

    for i in range(len(b_bbox_num)):
        bboxes = turn_into_bbox(b_x1[i], b_x2[i], b_x3[i], b_x4[i], b_y1[i], b_y2[i], b_y3[i], b_y4[i], b_bbox_num[i][0])
        j = now_num = b_bbox_num[i][0]
        while j < max_num:
            bboxes.append([0., 0., 0., 0.])
            j += 1
        batch_bboxes.append(bboxes)

        labels = [1 for j in range(now_num)]
        labels = labels + [0 for j in range(max_num - now_num)]
        batch_labels.append(labels)
    # print(bboxes)

    batch_labels = np.array(batch_labels)
    batch_bboxes = np.array(batch_bboxes)
    return batch_labels, batch_bboxes

def test_process():
    image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num = read_data(train=True)
    imgs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    labels = tf.placeholder(tf.int32, shape=[None, None])
    bboxes = tf.placeholder(tf.float32, shape=[None, None, 4])
    c_img, c_lab, c_bbox = get_processed_imgs(imgs, labels, bboxes)
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        b_image, b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num = \
                    sess.run([image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num])
        b_labels, b_bboxes = generate_batch_bboxes(b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num)
        re_img, re_lab, re_bbox=sess.run([c_img, c_lab, c_bbox],
             feed_dict={imgs:b_image[0:2], labels:b_labels[0:2], bboxes:b_bboxes[0:2]})

        result_img = Image.fromarray(np.uint8(re_img[0]))
        result_img.save('results/process/0.jpg')
        print(re_lab[0])
        print(re_bbox[0])




if __name__ == '__main__':
    #test_read()
    #generate_tfrecord()
    test_process()
    #image_data, shape, bboxes, difficults = process_image('/media/data2/hcx_data/ICDAR15-IncidentalSceneText/ch4_train_images/img_1.jpg')
    #print(bboxes)
