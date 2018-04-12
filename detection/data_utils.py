import os
from PIL import Image
import sys
import cv2
import random
import numpy as np
import tensorflow as tf
import tf_extended as tfe
import config_utils as config


slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

img_width = config.FLAGS.input_size_width
img_height = config.FLAGS.input_size_height


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
                                                   threshold=BBOX_CROP_OVERLAP,
                                                   assign_negative=False)
        return cropped_image, labels, bboxes, distort_bbox

def get_processed_imgs(images,
                       labels,
                       bboxes):
    ret_images = []
    ret_labels = []
    ret_bboxes = []
    for i im range(len(labels)):
        cropped_image, cropped_labels, cropped_bboxes = distorted_bounding_box_crop(images[0], labels[0], bboxes[0],
                                                                                    min_object_covered=0.25,
                                                                                    aspect_ratio_range=(0.6, 1.67))
        cropped_image = resize_image(cropped_image, (300,300),
                                     method=tf.image.ResizeMethod.BILINEAR,
                                     align_corners=False)
        ret_images.append(cropped_image)
        ret_labels.append(cropped_labels)
        ret_bboxes.append(cropped_bboxes)
        
    ret_images = np.array(ret_images)
    ret_labels = np.array(ret_labels)
    ret_bboxes = np.array(ret_bboxes)


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
    for root, dirs, filenames in os.walk(config.FLAGS.training_data_path):
        for file in filenames:
            if not file.endswith('.jpg'):
                continue
            files.append(os.path.join(root, file))
    print('{} training images in {}'.format(len(files), config.FLAGS.training_data_path))
    return files

def load_annotation(ano_path):
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


def process_image(filename):
    # image_data = tf.gfile.FastGFile(filename, 'rb').read()
    img = np.array(Image.open(filename))
    image_data = img.tostring()
    im = cv2.imread(filename)
    shape = im.shape

    anno_filename = filename.replace('.jpg', '.txt')
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


SAMPLE_PER_FILE = 200

def run(output_dir, shuffling=False, name='STV2K'):
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


# IMAGE_HEIGHT = 300
# IMAGE_WIDTH = 300

# filenames = '/media/data2/hcx_data/STV2KTF/STV2K_0000.tfrecord'
train_filenames = ['/media/data2/hcx_data/STV2KTF/STV2K_0000.tfrecord',
                   '/media/data2/hcx_data/STV2KTF/STV2K_0001.tfrecord',
                   '/media/data2/hcx_data/STV2KTF/STV2K_0002.tfrecord',
                   '/media/data2/hcx_data/STV2KTF/STV2K_0003.tfrecord',
                   '/media/data2/hcx_data/STV2KTF/STV2K_0004.tfrecord',
                   '/media/data2/hcx_data/STV2KTF/STV2K_0005.tfrecord',
                   '/media/data2/hcx_data/STV2KTF/STV2K_0006.tfrecord']
val_filenames = ['/media/data2/hcx_data/STV2KTF/STV2K_0003.tfrecord']


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
    run("/media/data2/hcx_data/STV2KTF/", shuffling=True)


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

        # for j in range(10):
            # im = Image.fromarray(np.uint8(img[i]))
            # im.save('out%d.jpg' % i)
            # print(img[j, :, :, :].shape)
            # print(x1s[j])
            # print(bbox_nums[j])
            # i += 1
            # print(i, bbox_nums[j])

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    test_read()
