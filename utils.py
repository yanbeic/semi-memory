from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from datasets import dataset_factory
from preprocessing import preprocessing_factory

FLAGS = tf.app.flags.FLAGS


def prepare_traindata(dataset_dir, batch_size):
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train', dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset=dataset, num_readers=4, shuffle=True)

    [image, label] = provider.get(['image', 'label'])

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocessing, is_training=True)
    image = image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)

    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=4,
                                            capacity=8 * batch_size, min_after_dequeue=4 * batch_size)
    return images, labels


def prepare_testdata(dataset_dir, batch_size):
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'test', dataset_dir)

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=1, shuffle=False)
    [image, label] = provider.get(['image', 'label'])

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocessing, is_training=False)
    image = image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)

    images, labels = tf.train.batch([image, label], batch_size=batch_size, num_threads=1,
                                    capacity=4 * batch_size, allow_smaller_final_batch=False)
    return images, labels


def config_lr(max_steps):
    if 'cifar' in FLAGS.dataset_name:
        # start to decay lr at the 250th epoch
        boundaries = [int(250.0 / 500.0 * max_steps)]
        values = [0.1]
    elif 'svhn' in FLAGS.dataset_name:
        # start to decay lr at the beginning: 0th epoch
        boundaries = [int(0 * max_steps)]
        values = [0.02]
    return boundaries, values


def linear_decay_lr(step, boundaries, values, max_steps):
    # decay learning rate linearly
    if 'svhn' in FLAGS.dataset_name:
        decayed_lr = (float(max_steps - (step + 1)) / float(max_steps)) * values[0]
    else:
        if step < boundaries[0]:
            decayed_lr = values[0]
        else:
            ratio = (float(max_steps - (step + 1)) / float(max_steps - boundaries[0]))
            decayed_lr = ratio * values[0]
    return decayed_lr


def cos_decay_lr(step, boundaries, values, max_steps):
    # decay learning rate with a cosine function
    if 'svhn' in FLAGS.dataset_name:
        ratio = 1. - (float(max_steps - (step + 1)) / float(max_steps))
        decayed_lr = np.cos(math.pi/2*ratio)* values[0]
    else:
        if step < boundaries[0]:
            decayed_lr = values[0]
        else:
            ratio = 1. - (float(max_steps - (step + 1)) / float(max_steps - boundaries[0]))
            decayed_lr = np.cos(math.pi/2*ratio)
            decayed_lr = decayed_lr * values[0]
    return decayed_lr


def sin_decay_lr(step, boundaries, values, max_steps):
    # decay learning rate with a sine function
    if 'svhn' in FLAGS.dataset_name:
        ratio = 1.- (float(max_steps - (step + 1)) / float(max_steps))
        decayed_lr = 1 - np.sin(math.pi/2*ratio)
        decayed_lr = decayed_lr * values[0]
    else:
        if step < boundaries[0]:
            decayed_lr = values[0]
        else:
            ratio = 1.- (float(max_steps - (step + 1)) / float(max_steps - boundaries[0]))
            decayed_lr = 1 - np.sin(math.pi/2*ratio)
            decayed_lr = decayed_lr * values[0]
    return decayed_lr


def decay_lr(step, boundaries, values, max_steps):
    # use cosine or sine learning rate decay schedule may further improve results
    if FLAGS.decay_lr_type == 'linear':
        decayed_lr = linear_decay_lr(step, boundaries, values, max_steps)
    elif FLAGS.decay_lr_type == 'cosine':
        decayed_lr = cos_decay_lr(step, boundaries, values, max_steps)
    elif FLAGS.decay_lr_type == 'sine':
        decayed_lr = sin_decay_lr(step, boundaries, values, max_steps)
    else:
        raise ValueError('decay_lr_type %s was not recognized.')
    return decayed_lr
