from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import images_to_tfrecords

tf.app.flags.DEFINE_string('dataset_dir', None, 'The dir of the raw image data.')
tf.app.flags.DEFINE_string('split_name', None, 'Either "train" or "test".')
tf.app.flags.DEFINE_string('dataset_name', None, 'The name of the dataset to convert.')
tf.app.flags.DEFINE_string('output_dir', None, 'The dir of the output data.')
tf.app.flags.DEFINE_integer('num_splits', None, 'The number of datasplits.')
tf.app.flags.DEFINE_integer('labels_per_class', None, 'The number of labeled sample per class.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')
    elif FLAGS.dataset_name in {'svhn','cifar10', 'cifar100'}:
        images_to_tfrecords.run(FLAGS.dataset_dir,
                                FLAGS.split_name,
                                FLAGS.dataset_name,
                                FLAGS.output_dir,
                                FLAGS.num_splits,
                                FLAGS.labels_per_class)
    else:
        raise ValueError('dataset_name [%s] was not recognized.' % FLAGS.dataset_dir)


if __name__ == '__main__':
    tf.app.run()
