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
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2" # config.FLAGS.gpu_list
model_dir='/home/hcxiao/Codes/YARR/detection/models/'
save_dir='/home/hcxiao/Codes/YARR/detection/models/stvnet-vgg/'
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



def train():

    with tf.Graph().as_default():
        # STVNet.redefine_params(img_width, img_height)

        image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num = data_utils.read_data(train=True)
        # g_bboxes = generate_batch_bboxes(x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num)
        # vimage, vx1_r, vx2_r, vx3_r, vx4_r, vy1_r, vy2_r, vy3_r, vy4_r, vbbox_num = data_utils.read_data(train=False)

        label = tf.placeholder(tf.int64, shape=[None, None], name='labels')
        bboxes = tf.placeholder(tf.float32, shape=[None, None, 4], name='bboxes')
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')

        anchors = STVNet.ssd_anchors_all_layers()
        predictions, localisations, logits, end_points = STVNet.model(inputs)
        gclasses, glocal, gscores = STVNet.tf_ssd_bboxes_batch_encode(label, bboxes, anchors, config.FLAGS.batch_size)
        #print('gclasses: ', gclasses)
        #print('glocal: ', glocal)
        #print('gscores: ', gscores)

        pos_loss, neg_loss, loc_loss, regular_loss= STVNet.ssd_losses(logits, localisations, gclasses, glocal, gscores)
        total_loss = pos_loss + neg_loss + loc_loss + regular_loss

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
        tf.summary.scalar("total_loss", total_loss)
        merged = tf.summary.merge_all()

        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if ckpt_path:
                # saver.restore(sess, save_dir + 'stvnet.ckpt')
                saver.restore(sess, ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

#            summary_writer = tf.summary.FileWriter('/home/hcxiao/STVLogs/tensorLog', sess.graph)
            batch_size = config.FLAGS.batch_size

            step = 301
            while_flag = True
            while(while_flag):

#                b_image, b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num = \
#                    sess.run([image, x1_r, x2_r, x3_r, x4_r, y1_r, y2_r, y3_r, y4_r, bbox_num])

#                b_labels, b_bboxes = generate_batch_bboxes(b_x1, b_x2, b_x3, b_x4, b_y1, b_y2, b_y3, b_y4, b_bbox_num)
                #print(b_labels.shape)
                #print(b_bboxes.shape)
                    
#                _, ploss, nloss, lcloss, gc, gl, gs, summary_str = sess.run([train_op, pos_loss, neg_loss, loc_loss, gclasses, glocal, gscores, merged],
#                                                                feed_dict={inputs: b_image, label: b_labels, bboxes: b_bboxes})

#                summary_writer.add_summary(summary_str, step)
#                summary_writer.flush()
                #print(gc[5])
                #print(gl[5])
                #print(gs[5])

#                tf.logging.info('%s: Step %d: PositiveLoss = %.2f' % (datetime.now(), step, ploss))#sum_ploss / (batch_size - flag)))
#                tf.logging.info('%s: Step %d: NegtiveLoss = %.2f' % (datetime.now(), step, nloss))#sum_nloss / (batch_size - flag)))
#                tf.logging.info('%s: Step %d: LocalizationLoss = %.2f' % (datetime.now(), step, lcloss))#sum_lcloss / (batch_size - flag)))

                if step % 100 == 0:
                    saver.save(sess, save_dir + 'stvnet.ckpt', global_step=step)
                step += 1

                while_flag = False


            coord.request_stop()
            coord.join(threads)


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


# useless
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
