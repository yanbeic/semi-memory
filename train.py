from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import semi_learner
import sup_learner

# training settings
tf.app.flags.DEFINE_boolean('semi', True, 'Train in semi-supervised learning mode.')
tf.app.flags.DEFINE_float('label_ratio', 0.5, 'The ratio of labelled data.')
tf.app.flags.DEFINE_integer('num_epochs', 500, "the number of epochs for training")
tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs to use.')
tf.app.flags.DEFINE_float('ema_decay', 0.9999, 'If None, then not used.')
tf.app.flags.DEFINE_string('train_dir', '/tmp/tfmodel/', 'Where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_float('gpu_memory', None, 'The memory percentage.')
tf.app.flags.DEFINE_string('decay_lr_type', 'linear', 'Three types: linear, cosine, sine.')

# dataset
tf.app.flags.DEFINE_integer('num_train_l', 10000, 'The number of labelled samples.')
tf.app.flags.DEFINE_integer('num_train_u', 40000, 'The number of unlabelled samples.')
tf.app.flags.DEFINE_integer('num_classes', 100, 'The number of classes in training data.')
tf.app.flags.DEFINE_string('dataset_name', 'cifar100', 'The name of the dataset.')
tf.app.flags.DEFINE_string('dataset_dir_l', None, 'Where the labelled data is stored.')
tf.app.flags.DEFINE_string('dataset_dir_u', None, 'Where the unlabelled data is stored.')

# preprocessing
tf.app.flags.DEFINE_integer('batch_size', 100, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image size.')
tf.app.flags.DEFINE_string('preprocessing', 'cifar', 'The type of the preprocessing to use.')

# network
tf.app.flags.DEFINE_string('model_name', 'convnet', 'The name of the architecture to train.')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float('smoothing', 0.001, 'The smoothing factor in each class label.')

# memory module
tf.app.flags.DEFINE_float('eta', 0.5, 'The default memory update rate.')
tf.app.flags.DEFINE_float('weight_u', 1.0, 'The weight on unsupervised loss.')
tf.app.flags.DEFINE_string('feature_name', 'AvgPool', 'Name of the feature layer.')
tf.app.flags.DEFINE_integer('feature_dim', 128, 'Dim of the feature layer.')

FLAGS = tf.app.flags.FLAGS


def main(_):

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    if FLAGS.semi:
        semi_learner.train()
    else:
        sup_learner.train()


if __name__ == '__main__':
    tf.app.run()
