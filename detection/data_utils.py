import os
from PIL import Image
import sys
import cv2
import random
import numpy as np
import tensorflow as tf
import config_utils as config


slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300

filenames = '/media/data2/hcx_data/STV2KTF/STV2K_0000.tfrecord'

def read_and_decode():
    filename_queue = tf.train.string_input_producer(['/media/data2/hcx_data/STV2KTF/STV2K_0000.tfrecord',
                                                     '/media/data2/hcx_data/STV2KTF/STV2K_0001.tfrecord',
                                                     '/media/data2/hcx_data/STV2KTF/STV2K_0002.tfrecord',
                                                     '/media/data2/hcx_data/STV2KTF/STV2K_0003.tfrecord',
                                                     '/media/data2/hcx_data/STV2KTF/STV2K_0004.tfrecord',
                                                     '/media/data2/hcx_data/STV2KTF/STV2K_0005.tfrecord',
                                                     '/media/data2/hcx_data/STV2KTF/STV2K_0006.tfrecord'],
                                                    num_epochs=10)

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
    print(x1, bbox_num)
    bbox_shape = tf.stack([bbox_num[0]])
    x1 = tf.reshape(x1, bbox_shape)

    image_shape = tf.stack([height[0], width[0], 3])
    image = tf.reshape(image, image_shape)
    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
    # resize_image = tf.image.resize_image_with_crop_or_pad(image=image,
    #                                                       target_height=IMAGE_HEIGHT,
    #                                                       target_width=IMAGE_WIDTH)
    resize_image = tf.image.resize_images(image, size=[IMAGE_HEIGHT, IMAGE_WIDTH])

    resize_ratio_x = tf.cast(IMAGE_WIDTH / width[0], tf.float32)
    x1_r = x1 * resize_ratio_x

    # image.set_shape([height[0], width[0], 3])
    images, x1s, x1_rs, bbox_nums = tf.train.batch([resize_image, x1, x1_r, bbox_num],
                                    batch_size=10,
                                    capacity=30,
                                    num_threads=2,
                                    # min_after_dequeue=10,
                                    dynamic_pad=True)
    return images, x1s, x1_rs, bbox_nums

# if __name__ == '__main__':
    # get_images()
    # polys = load_annotation('/home/hcxiao/Datasets/STV2k/stv2k_train/STV2K_tr_0001.txt')
    # print(polys)
    # data_generator = get_batch(4, 32, 8)
    # data = next(data_generator)
    # run("/media/data2/hcx_data/STV2KTF/", shuffling=True)
    # test_read()


def test_read():
    image, x1, x1_r, bbox_num = read_and_decode()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img, x1s, x1_rs, bbox_nums = sess.run([image, x1, x1_r, bbox_num])
        print(x1s[0])
        print(x1_rs[0])
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
