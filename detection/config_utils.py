import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('training_data_path','/media/data2/hcx_data/STV2K/stv2k_train/',
                           'the path to training dataset')
tf.app.flags.DEFINE_string('gpu_list', '1',
                           'the list of gpu to use')
tf.app.flags.DEFINE_integer('text_scale', 512, '')
