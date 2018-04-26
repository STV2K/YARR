import os
import cv2
import math
import numpy as np
import tensorflow as tf
import config_utils as config
import data_utils as data_utils
import tf_extended as tfe

from nets import STVNet
from PIL import Image
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_dir='/home/hcxiao/Codes/YARR/detection/models/stvnet/'
STV2K_Path = '/media/data1/hcxiao/STV2K/stv2k_test/'
ICDAR_Path='/media/data1/hcxiao/ICDAR15-IST/test_images/'
img_name = 'STV2K_ts_0575.jpg' #'img_243.jpg'
PATH = STV2K_Path

test_all_path = ICDAR_Path
generate_pic_path = './results/icdar/images-400*400/'
generate_txt_path = './results/icdar/texts/'
generate_threshold = 0.05

img_width = config.FLAGS.input_size_width
img_height = config.FLAGS.input_size_height
input_size = (img_width, img_height)
ckpt_path = config.FLAGS.ckpt_path
# gpu_list = config.FLAGS.gpu_list.split(',')
# gpus = [int(gpu_list[i]) for i in range(len(gpu_list))]

def get_image(img_path):
    im = Image.open(img_path)
    ori_width, ori_height = im.size
    im = im.resize(input_size)
    im = np.array(im)

    return im, ori_width, ori_height

def convert_poly_to_bbox(polys, ori_width, ori_height):
    bboxes = []
    print(ori_width, ori_height)
    for poly in polys:
        (x1, y1, x2, y2, x3, y3, x4, y4) = poly
        x = [x1, x2, x3, x4]
        y = [y1, y2, y3, y4]
        xmin = min(x) / ori_width
        xmax = max(x) / ori_width
        ymin = min(y) / ori_height
        ymax = max(y) / ori_height

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0

        bbox = [ymin, xmin, ymax, xmax]
        bboxes.append(bbox)
    return bboxes

def test(img_name):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # STVNet.redefine_params(img_width, img_height)

        im, ori_width, ori_height = get_image(PATH + img_name)
        #polys,_ = data_utils.load_annotation(STV2K_Path + img_name.replace('.jpg', '.txt'))

        label = tf.placeholder(tf.int64, shape=[None], name='labels')
        bboxes = tf.placeholder(tf.float32, shape=[None, 4], name='bboxes')
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')
        b_gdifficults = tf.zeros(tf.shape(label), dtype=tf.int64)

        # anchors = STVNet.ssd_anchors_all_layers()
        # predictions, localisations, logits, end_points = STVNet.model(inputs, is_training=True)
        params = STVNet.redefine_params(STVNet.DetectionNet.default_params, config.FLAGS.input_size_width, config.FLAGS.input_size_height)
        detection_net = STVNet.DetectionNet(params)

        anchors = detection_net.anchors() #STVNet.ssd_anchors_all_layers()
        predictions, localisations, logits, pre_angles, end_points = detection_net.model(inputs) #STVNet.model(inputs)

        # with tf.device('/device:CPU:0'):
        pre_locals = STVNet.tf_ssd_bboxes_decode(localisations, anchors,
                                                 detection_net.params.anchor_steps,
                                                 detection_net.params.img_shape,
                                                 detection_net.params.prior_scaling,
                                                 offset=True, scope='bboxes_decode')
        pre_scores, pre_bboxes, p_angles = STVNet.detected_bboxes(predictions, pre_locals, pre_angles,
                                                        select_threshold=config.FLAGS.select_threshold,
                                                        nms_threshold=config.FLAGS.nms_threshold,
                                                        clipping_bbox=None,
                                                        top_k=config.FLAGS.select_top_k,
                                                        keep_top_k=config.FLAGS.keep_top_k)

        saver = tf.train.Saver()
        # gpu_fraction = 0.5
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            if ckpt_path:
                saver.restore(sess, ckpt_path)
            else:
                saver.restore(sess, model_dir + 'stvnet.ckpt-18100')

            #gt_bboxes = convert_poly_to_bbox(polys, ori_width, ori_height)
            #gt_labels = [1 for i in range(len(gt_bboxes))]

            pre_s, pre_box, pre_an = sess.run([pre_scores, pre_bboxes, p_angles],
                                      feed_dict={inputs: [im]})
            
            img = Image.open(PATH + img_name)
            img = np.array(img)

            # img = np.copy(im)
            bboxes_draw_on_img(img, pre_s[1][0], pre_box[1][0], pre_an[1][0], generate_threshold, (31, 119, 180))
            # fig = plt.figure(figsize=(12, 12))
            # plt.imshow(img)
            result_img = Image.fromarray(np.uint8(img))
            result_img.save('results/result-angle-test-' + str(config.FLAGS.input_size_width) + '-' + ckpt_path.split('-')[-1] + '-' + img_name)


def test_all(dir_path):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():

        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')

        #anchors = STVNet.ssd_anchors_all_layers()
        #predictions, localisations, logits, end_points = STVNet.model(inputs)
        params = STVNet.redefine_params(STVNet.DetectionNet.default_params, config.FLAGS.input_size_width, config.FLAGS.input_size_height)
        detection_net = STVNet.DetectionNet(params)

        anchors = detection_net.anchors()
        predictions, localisations, logits, pre_angles, end_points = detection_net.model(inputs)

        pre_locals = STVNet.tf_ssd_bboxes_decode(localisations, anchors,
                                                 detection_net.params.anchor_steps,
                                                 detection_net.params.img_shape,
                                                 detection_net.params.prior_scaling,
                                                 offset=True, scope='bboxes_decode')
        pre_scores, pre_bboxes, p_angles = STVNet.detected_bboxes(predictions, pre_locals, pre_angles,
                                                        select_threshold=config.FLAGS.select_threshold,
                                                        nms_threshold=config.FLAGS.nms_threshold,
                                                        clipping_bbox=None,
                                                        top_k=config.FLAGS.select_top_k,
                                                        keep_top_k=config.FLAGS.keep_top_k)

        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            if ckpt_path:
                saver.restore(sess, ckpt_path)
            else:
                saver.restore(sess, model_dir + 'stvnet.ckpt-18100')

            for root, dirnames, filenames in os.walk(dir_path):
                for filename in filenames:
                    if not filename.endswith('.jpg'):
                        continue

                    im, ori_width, ori_height = get_image(dir_path + filename)
                    pre_s, pre_box, pre_an = sess.run([pre_scores, pre_bboxes, p_angles],
                                              feed_dict={inputs: [im]})

                    img = Image.open(dir_path + filename)
                    img = np.array(img)
                    bboxes_draw_on_img(img, pre_s[1][0], pre_box[1][0], pre_an[1][0], generate_threshold, (31, 119, 180))
                    result_img = Image.fromarray(np.uint8(img))
                    result_img.save(generate_pic_path + filename)
#                    txt_generator(filename, pre_s[1][0], pre_box[1][0], generate_threshold, ori_width, ori_height)
                    print(filename + ' finished.')


def txt_generator(filename, scores, bboxes, threshold, width, height):
    txt_save_path = os.path.join(generate_txt_path, 'res_' + filename.split('.')[0]+'.txt')
    txt_file = open(txt_save_path, 'wt')

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        score = scores[i]
        if score < threshold:
            continue
        ymin = str(int(bbox[0]*height))
        xmin = str(int(bbox[1]*width))
        ymax = str(int(bbox[2]*height))
        xmax = str(int(bbox[3]*width))
        result_str = xmin+','+ymin+','+xmax+','+ymin+','+xmax+','+ymax+','+xmin+','+ymax+'\r\n'
        txt_file.write(result_str)

    txt_file.close()


        
def bboxes_draw_on_img(img, scores, bboxes, angles, threshold, color, thickness=5):
    shape = img.shape
    for i in range(len(bboxes)):
        if(scores[i] > threshold):
            color = (255, 255, 0)
        else:
            color = (31, 119, 180)
            continue
        bbox = bboxes[i]
        # Draw bounding box...
#        if bbox[0] < 0:
#            bbox[0] = 0
#        if bbox[1] < 0:
#            bbox[1] = 0
#        if bbox[2] > 1:
#            bbox[2] = 1
#        if bbox[3] > 1:
#            bbox[3] = 1
        bbox = np.clip(bbox, 0, 1)
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
#        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)

        # Draw rbox
        angle = angles[i] / np.pi * 180
        #width = bbox[3] - bbox[1]
        height = p2[0] - p1[0]
        dx = int(height / 2. * math.cos(angles[i]))
        dy = int(height / 2. * math.sin(angles[i]))
        if angle < 0:
            cor1 = (p1[1] + dx, p2[0] - dy)
            cor2 = (p1[1] - dx, p2[0] + dy)
            cor3 = (p2[1] - dx, p1[0] + dy)
            cor4 = (p2[1] + dx, p1[0] - dy)
        else:
            cor1 = (p1[1] + dx, p1[0] - dy)
            cor2 = (p1[1] - dx, p1[0] + dy)
            cor3 = (p2[1] - dx, p2[0] + dy)
            cor4 = (p2[1] + dx, p2[0] - dy)
        rbox = np.array([cor1, cor2, cor3, cor4])
        #cv2.polylines(img, [rbox], True, (0, 255, 0), 3)
        cv2.drawContours(img, [rbox], -1, (255, 255, 0), 5)

        # Draw text...
        s = '%.3f' % (scores[i]) #, angles[i] / np.pi * 180)
        p1 = (p1[0]-5, p1[1])
#        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, color, 2)


if __name__ == '__main__':
#    test(img_name)
    test_all(test_all_path)
    

