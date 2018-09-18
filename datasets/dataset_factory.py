from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import svhn
from datasets import cifar10
from datasets import cifar100

FLAGS = tf.app.flags.FLAGS


datasets_map = \
    {
        'svhn': svhn,
        'cifar10': cifar10,
        'cifar100': cifar100,
    }


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_split(
        split_name,
        dataset_dir,
        file_pattern,
        reader)
