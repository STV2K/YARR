import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('training_data_path','/media/data2/hcx_data/STV2K/stv2k_train/',
                           'the path to training dataset')
tf.app.flags.DEFINE_string('gpu_list', '1',
                           'the list of gpu to use')
tf.app.flags.DEFINE_integer('text_scale', 512, '')
tf.app.flags.DEFINE_integer('batch_size', 16, '')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, '')
tf.app.flags.DEFINE_float('select_threshold', 0.01, 'Selection threshold.')
tf.app.flags.DEFINE_integer('select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer('keep_top_k', 200, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_float('nms_threshold', 0.45, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float('matching_threshold', 0.5, 'Matching threshold with groundtruth objects.')
