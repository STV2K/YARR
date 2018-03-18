import os
import PIL
import sys
import cv2
import random
import numpy as np
import tensorflow as tf
import config_utils as config


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
    print(filename)
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
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
              'image/shape': int64_feature(shape),
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


def get_tf_data(dataset_dir, file_pattern, reader=None):


    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
                        'image/height': tf.FixedLenFeature([1], tf.int64),
                        'image/width': tf.FixedLenFeature([1], tf.int64),
                        'image/channels': tf.FixedLenFeature([1], tf.int64),
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
                        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg')
    }

    items_to_handlers = {
                        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format')
                        # 'shape': slim.tfexample_decoder.Tensor('image/shape'),
                        'object/x1': slim.tfexample_decoder.Tensor('image/object/bbox/x1'),
                        'object/y1': slim.tfexample_decoder.Tensor('image/object/bbox/y1'),
                        'object/x2': slim.tfexample_decoder.Tensor('image/object/bbox/x2'),
                        'object/y2': slim.tfexample_decoder.Tensor('image/object/bbox/y2'),
                        'object/x3': slim.tfexample_decoder.Tensor('image/object/bbox/x3'),
                        'object/y3': slim.tfexample_decoder.Tensor('image/object/bbox/y3'),
                        'object/x4': slim.tfexample_decoder.Tensor('image/object/bbox/x4'),
                        'object/y4': slim.tfexample_decoder.Tensor('image/object/bbox/y4'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    return slim.dataset.Dateset(
                                data_source=file_pattern,
                                reader=reader,
                                decoder=decoder,
                                num_samples=1215,
                                )


if __name__ == '__main__':
    # get_images()
    # polys = load_annotation('/home/hcxiao/Datasets/STV2k/stv2k_train/STV2K_tr_0001.txt')
    # print(polys)
    # data_generator = get_batch(4, 32, 8)
    # data = next(data_generator)
    # run("/media/data2/hcx_data/STV2KTF", shuffling=True)

    data = get_tf_data(config.FLAGS.training_data_path, '/media/data2/hcx_data/STV2KTF/STV2K_0000.tfrecord')
    provider = slim.dataset_data_provider.DatasetDataProvider(
                    data,
                    num_readers=4,
                    common_queue_capacity=20 * 32,
                    common_queue_min=10 * 32,
                    shuffle=True)
    [image, x1, y1] = procider.get(['image', 'object/x1', 'object/y1']) # , x2, y2, x3, y3, x4, y4


