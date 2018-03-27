import os
import numpy as np
import tensorflow as tf
import config_utils as config
import data_utils as data_utils

from nets import STVNet

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # config.FLAGS.gpu_list


def turn_into_bbox(x1, x2, x3, x4, y1, y2, y3, y4, num):
    bboxes = []
    for i in range(num):
        x = [x1[i], x2[i], x3[i], x4[i]]
        y = [y1[i], y2[i], y3[i], y4[i]]
        xmin = min(x) / 300.0
        xmax = max(x) / 300.0
        ymin = min(y) / 300.0
        ymax = max(y) / 300.0

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0

        bbox = [ymin, xmin, ymax, xmax]
        bboxes.append(bbox)

    return bboxes


def generate_batch_bboxes(b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num):
    batch_bboxes = []
    for i in range(len(b_bbox_num)):
        bboxes = turn_into_bbox(b_x1[i], b_x2[i], b_x3[i], b_x4[i], b_y1[i], b_y2[i], b_y3[i], b_y4[i], b_bbox_num[i][0])
        batch_bboxes.append(bboxes)
    # print(bboxes)

    return batch_bboxes



def main():

    image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num = data_utils.read_and_decode()
    
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')
    predictions, localisations, logits, end_points = STVNet.model(inputs)

    anchors = STVNet.ssd_anchors_all_layers()
    label = tf.placeholder(tf.int64, shape=[None], name='labels')
    bboxes = tf.placeholder(tf.float32, shape=[None, 4], name='bboxes')
    gc, gl, gs = STVNet.tf_ssd_bboxes_encode(label, bboxes, anchors)

    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        b_image, b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num = \
            sess.run([image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num])

        b_bboxes = generate_batch_bboxes(b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num)


        pres, locs, f_score, f_geo = sess.run([predictions, localisations, logits, end_points], feed_dict={inputs: [b_image[0]]})

        print('block 1 shape: ',  f_geo['resnet_v1_50/block1'].shape)
        print('block 2 shape: ',  f_geo['resnet_v1_50/block2'].shape)
        print('block 3 shape: ',  f_geo['resnet_v1_50/block3'].shape)
        print('block 4 shape: ',  f_geo['block4'].shape)
        print('block 5 shape: ',  f_geo['block5'].shape)
        print('block 6 shape: ',  f_geo['block6'].shape)
        print('block 7 shape: ',  f_geo['block7'].shape)
        print('block 8 shape: ',  f_geo['block8'].shape)
        print('block 9 shape: ',  f_geo['block9'].shape)
        print('f_score[0].shape : ', f_score[0].shape)
        print('box_num : ', b_bbox_num[0])

        # anchors = STVNet.ssd_anchors_all_layers()
        # print(len(anchors))
        # print(anchors[4])
        # print(anchors[5])

        labels = [1 for i in range(b_bbox_num[0][0])]
        # gclasses, glocal, gscores = STVNet.tf_ssd_bboxes_encode(labels, b_bboxes[0], anchors)
        gclasses, glocal, gscores = sess.run([gc, gl, gs], feed_dict={label: labels, bboxes: b_bboxes[0]})

        # print('gclasses0: ', gclasses[0])
        # print('glocal4: ', glocal[4])
        # print(len(glocal[4]), len(glocal[4][0]), len(glocal[4][0][0]), len(glocal[4][0][0][0]))
        # print('gscores: ', gscores)

        # merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('/home/hcxiao/STVLogs', sess.graph)

        STVNet.ssd_losses(f_score, locs, gclasses, glocal, gscores)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
