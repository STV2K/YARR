import os
import numpy as np
import tensorflow as tf
import config_utils as config
import data_utils as data_utils

from tensorflow.python import pywrap_tensorflow
from datetime import datetime
from nets import STVNet

tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # config.FLAGS.gpu_list
log_dir='/home/hcxiao/Logs/tensorlog'
model_dir='/home/hcxiao/Codes/YARR/detection/models/'
save_dir='/home/hcxiao/Codes/xhc_dev/YARR/detection/models/angles/1'
model_name='VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt' # .data-00000-of-00001'

img_width = config.FLAGS.input_size_width
img_height = config.FLAGS.input_size_height
ckpt_path = config.FLAGS.ckpt_path

def get_model_data(model):
    ckpt_path = os.path.join(model_dir, model)
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_map = reader.get_variable_to_shape_map()

    # this loads variables in session within the code
    # var_list = tf.global_variables()
    # for key in var_map:
        # print('tensor_name: ', key)
        # print(reader.get_tensor(key))
    return var_map, reader

def get_angle(poly):
    poly = np.array(poly)
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        return 0.
    else:
        p_lowest_right = (p_lowest - 1) % 4
        # p_lowest_left = (p_lowest + 1) % 4
        divide_num = (poly[p_lowest][0] - poly[p_lowest_right][0])
        if divide_num == 0:
            angle = 0
        else:
            angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1]) / divide_num)

        if angle / np.pi * 180 > 45:
            return -(np.pi / 2 - angle)
        else:
            return angle

def turn_into_bbox(x1, x2, x3, x4, y1, y2, y3, y4, num):
    bboxes = []
    angles = []
    for i in range(num):
        x = [x1[i], x2[i], x3[i], x4[i]]
        y = [y1[i], y2[i], y3[i], y4[i]]
        xmin = min(x) / img_width
        xmax = max(x) / img_width
        ymin = min(y) / img_height
        ymax = max(y) / img_height

        # if xmin < 0:
        #     xmin = 0
        # if ymin < 0:
        #     ymin = 0
        # if xmax > 1:
        #     xmax = 1
        # if ymax > 1:
        #     ymax = 1
        edge = [
                (x2[i] - x1[i]) * (y2[i] + y1[i]),
                (x3[i] - x2[i]) * (y3[i] + y2[i]),
                (x4[i] - x3[i]) * (y4[i] + y3[i]),
                (x1[i] - x4[i]) * (y1[i] + y4[i]),
                ]
        area =  np.sum(edge) / 2.

        if area > 0:
            angle = get_angle([[x1[i], y1[i]], [x2[i], y2[i]], [x3[i], y3[i]], [x4[i], y4[i]]])
        else:
            angle = get_angle([[x1[i], y1[i]], [x4[i], y4[i]], [x3[i], y3[i]], [x2[i], y2[i]]])

        bbox = np.clip([ymin, xmin, ymax, xmax], 0.0, 1.0)
        bboxes.append(bbox)
        angles.append(angle)

    return bboxes, angles


def generate_batch_bboxes(b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num):
    batch_bboxes = []
    batch_angles = []
    batch_labels = []
    max_num = 0
    for i in range(len(b_bbox_num)):
        if max_num < b_bbox_num[i][0]:
            max_num = b_bbox_num[i][0]

    for i in range(len(b_bbox_num)):
        bboxes, angles = turn_into_bbox(b_x1[i], b_x2[i], b_x3[i], b_x4[i], b_y1[i], b_y2[i], b_y3[i], b_y4[i], b_bbox_num[i][0])
        j = now_num = b_bbox_num[i][0]
        while j < max_num:
            bboxes.append([0., 0., 0., 0.])
            angles.append(0.)
            j += 1
        batch_bboxes.append(bboxes)
        batch_angles.append(angles)

        labels = [1 for j in range(now_num)]
        labels = labels + [0 for j in range(max_num - now_num)]
        batch_labels.append(labels)
    # print(bboxes)

    batch_labels = np.array(batch_labels)
    batch_bboxes = np.array(batch_bboxes)
    batch_angles = np.array(batch_angles)
    return batch_labels, batch_bboxes, batch_angles



def train():

    with tf.Graph().as_default():
        # STVNet.redefine_params(img_width, img_height)

        image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num = data_utils.read_data(train=True)
        
        label = tf.placeholder(tf.int64, shape=[None, None], name='labels')
        bboxes = tf.placeholder(tf.float32, shape=[None, None, 4], name='bboxes')
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')
        angles = tf.placeholder(tf.float32, shape=[None, None], name='angles')

        params = STVNet.redefine_params(STVNet.DetectionNet.default_params, config.FLAGS.input_size_width, config.FLAGS.input_size_height)
        detection_net = STVNet.DetectionNet(params)

        anchors = detection_net.anchors() #STVNet.ssd_anchors_all_layers()
        predictions, localisations, logits, pre_angles, end_points = detection_net.model(inputs) #STVNet.model(inputs)
        gclasses, glocal, gscores, gangles = STVNet.tf_ssd_bboxes_batch_encode(label, bboxes, angles, anchors,
                                                                      config.FLAGS.batch_size,
                                                                      detection_net.params.anchor_steps,
                                                                      detection_net.params.img_shape)

        pos_loss, neg_loss, loc_loss, regular_loss, angle_loss= STVNet.ssd_losses(logits, localisations, pre_angles, gclasses, glocal, gscores, gangles)
        total_loss = pos_loss + neg_loss + loc_loss + regular_loss + angle_loss

        global_step = tf.Variable(0, name='global_step', trainable=False)
        initial_learning_rate = config.FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=304,
                                                   decay_rate=0.94,
                                                   staircase=True,
                                                   name='exponential_decay_learning_rate')
        tf.summary.scalar("learning_rate", learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate,
                                           beta1=0.9,
                                           beta2=0.999,
                                           epsilon=1.0)

        train_op = optimizer.minimize(total_loss, global_step=global_step)
        tf.summary.scalar("pos_loss", pos_loss)
        tf.summary.scalar("neg_loss", neg_loss)
        tf.summary.scalar("loc_loss", loc_loss)
        tf.summary.scalar("regular_loss", regular_loss)
        tf.summary.scalar("angle_loss", angle_loss)
        tf.summary.scalar("total_loss", total_loss)
        merged = tf.summary.merge_all()

        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if ckpt_path:
                saver.restore(sess, ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            batch_size = config.FLAGS.batch_size

            step = 1
            while_flag = True
            while(while_flag):

                b_image, b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num = \
                    sess.run([image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num])

                b_labels, b_bboxes, b_angles = generate_batch_bboxes(b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num)
                #print(b_labels.shape)
                #print(b_bboxes.shape)
                    
                _, ploss, nloss, lcloss, aloss, summary_str = sess.run([train_op, pos_loss, neg_loss, loc_loss, angle_loss, merged],
                                                                feed_dict={inputs: b_image, label: b_labels, bboxes: b_bboxes, angles:b_angles})

                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                tf.logging.info('%s: Step %d: PositiveLoss = %.2f' % (datetime.now(), step, ploss))#sum_ploss / (batch_size - flag)))
                tf.logging.info('%s: Step %d: NegtiveLoss = %.2f' % (datetime.now(), step, nloss))#sum_nloss / (batch_size - flag)))
                tf.logging.info('%s: Step %d: LocalizationLoss = %.2f' % (datetime.now(), step, lcloss))#sum_lcloss / (batch_size - flag)))
                tf.logging.info('%s: Step %d: LocalizationAngleLoss = %.2f' % (datetime.now(), step, aloss))

                if step % 200 == 0:
                    saver.save(sess, save_dir + 'stvnet.ckpt', global_step=step)
                step += 1

#                while_flag = False


            coord.request_stop()
            coord.join(threads)


# useless
def validate(saver):

    # ckpt_dir = save_dir
    # last_checkpoint = saver.latest_checkpoint(ckpt_dir)

    # validation
    if step % 50 == 0:
        val_b_image, val_b_x1, val_b_x2, val_b_x3, val_b_x4, val_b_y1, val_b_y2, val_b_y3, val_b_y4, val_b_bbox_num = \
            sess.run([vimage, vx1_r, vx2_r, vx3_r, vx4_r, vy1_r, vy2_r, vy3_r, vy4_r, vbbox_num])
        val_bboxes = generate_batch_bboxes(val_b_x1, val_b_x2, val_b_x3, val_b_x4,
                                           val_b_y1, val_b_y2, val_b_y3, val_b_y4,
                                           val_b_bbox_num)

        for i in range(batch_size):
            labels = [1 for i in range(b_bbox_num[i][0])]

            _, ploss, nloss, lcloss, summary_str = sess.run([train_op, pos_loss, neg_loss, loc_loss, merged],
                                                                    feed_dict={inputs: [b_image[i]], label: labels, bboxes: b_bboxes[i]})


def change_paras():
    var_ssd, reader_ssd = get_model_data(model_name)
    var_stv, reader_stv = get_model_data('stvnet/stvnet.ckpt')
    var_list = {}

    i = 6
    while i < 10:
        b = i + 2
        print("trsforming stvnet/block%i" % i, "  to ssd_300_vgg/block%i" % b)
        var_stv['stvnet/block%i_box/conv_loc/weights' % i] = reader_ssd.get_tensor('ssd_300_vgg/block%i_box/conv_loc/weights' % b)
        var_stv['stvnet/block%i_box/conv_cls/weights' % i] = reader_ssd.get_tensor('ssd_300_vgg/block%i_box/conv_cls/weights' % b)
        var_stv['stvnet/block%i/conv1x1/weights' % i] = reader_ssd.get_tensor('ssd_300_vgg/block%i/conv1x1/biases' % b)
        var_stv['stvnet/block%i/conv3x3/weights' % i] = reader_ssd.get_tensor('ssd_300_vgg/block%i/conv3x3/biases' % b)
        i += 1

    var_stv['stvnet/conv6/weights'] = reader_ssd.get_tensor('ssd_300_vgg/conv6/weights')
    var_stv['stvnet/conv7/weights'] = reader_ssd.get_tensor('ssd_300_vgg/conv7/weights')

    var_list['stvnet/conv6/weights'] = reader_ssd.get_tensor('ssd_300_vgg/conv6/weights')
    var_list['stvnet/conv7/weights'] = reader_ssd.get_tensor('ssd_300_vgg/conv7/weights')

    return var_stv, var_list


if __name__ == '__main__':
    train()
    # get_model_data(model_name)
