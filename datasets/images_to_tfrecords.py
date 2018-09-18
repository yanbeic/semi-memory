from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import utils

# The height and width of each image.
_IMAGE_SIZE = 32


def _write_to_tfrecord(filenames, labels, tfrecord_writer):
    num_images = len(filenames)
    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder)
        with tf.Session('') as sess:
            for i in range(num_images):
                sys.stdout.write('\r>> Reading images %d/%d' % (i + 1, num_images))
                sys.stdout.flush()
                image_path = filenames[i]
                image = Image.open(image_path)
                label = labels[i]
                png_string = sess.run(encoded_image, feed_dict={image_placeholder: image})
                example = utils.image_to_tfexample(png_string, 'png', _IMAGE_SIZE, _IMAGE_SIZE, label)
                tfrecord_writer.write(example.SerializeToString())


def _get_testdata_filenames_and_labels(dataset_dir):
    classes = os.listdir(dataset_dir)
    filenames = []
    labels = []
    for i in range(len(classes)):
        classpath = os.path.join(os.getcwd(), dataset_dir, classes[i])
        imgpaths = os.listdir(classpath)
        imgpaths = [os.path.join(classpath, substring) for substring in imgpaths]
        filenames.append(imgpaths)
        labels.append(i * np.ones(len(imgpaths), dtype=np.int64))
    filenames = np.concatenate(filenames, 0)
    labels = np.concatenate(labels, 0)
    return filenames, labels


def _shuffle_data(filenames, labels):
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]
    return filenames, labels


def _get_traindata_filenames_and_labels(dataset_dir, labels_per_class):
    classes = os.listdir(dataset_dir)
    filenames_l = []
    filenames_u = []
    labels_l = []
    labels_u = []
    for i in range(len(classes)):
        classpath = os.path.join(os.getcwd(), dataset_dir, classes[i])
        imgpaths = os.listdir(classpath)
        imgpaths = [os.path.join(classpath, substring) for substring in imgpaths]
        random.seed(12345)
        random.shuffle(imgpaths)
        imgpaths_l = imgpaths[:labels_per_class]
        imgpaths_u = imgpaths[labels_per_class:]
        filenames_l.append(imgpaths_l)
        filenames_u.append(imgpaths_u)
        labels_l.append(i * np.ones(labels_per_class, dtype=np.int64))
        labels_u.append(-1 * np.ones(len(imgpaths) - labels_per_class, dtype=np.int64))
    filenames_l = np.concatenate(filenames_l, 0)
    filenames_u = np.concatenate(filenames_u, 0)
    labels_l = np.concatenate(labels_l, 0)
    labels_u = np.concatenate(labels_u, 0)
    # shuffle traindata completely
    filenames_l, labels_l = _shuffle_data(filenames_l, labels_l)
    filenames_u, labels_u = _shuffle_data(filenames_u, labels_u)
    return filenames_l, labels_l, filenames_u, labels_u


def _make_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def _get_tfrecords_names(output_dir, dataset_name, split_name):
    tfrecords_names = '%s/%s_%s.tfrecord' % (output_dir, dataset_name, split_name)
    _make_dir(output_dir)
    return tfrecords_names


def run(dataset_dir, split_name, dataset_name, output_dir, num_splits, labels_per_class):

    if split_name == 'test':
        tfrecords_names = _get_tfrecords_names(output_dir, dataset_name, split_name)
        filenames, labels = _get_testdata_filenames_and_labels(dataset_dir)
        with tf.python_io.TFRecordWriter(tfrecords_names) as tfrecord_writer:
            _write_to_tfrecord(filenames, labels, tfrecord_writer)
    else:
        for i in range(num_splits):
            output_sub_dir_l = os.path.join(output_dir, 'split'+str(i+1), 'labeled')
            tfrecords_names_l = _get_tfrecords_names(output_sub_dir_l, dataset_name, split_name)

            output_sub_dir_u = os.path.join(output_dir, 'split'+str(i+1), 'unlabeled')
            tfrecords_names_u = _get_tfrecords_names(output_sub_dir_u, dataset_name, split_name)

            filenames_l, labels_l, filenames_u, labels_u = \
                _get_traindata_filenames_and_labels(dataset_dir, labels_per_class)

            print('\nConverting the labelled data.')
            with tf.python_io.TFRecordWriter(tfrecords_names_l) as tfrecord_writer:
                _write_to_tfrecord(filenames_l, labels_l, tfrecord_writer)

            print('\nConverting the unlabelled data.')
            with tf.python_io.TFRecordWriter(tfrecords_names_u) as tfrecord_writer:
                _write_to_tfrecord(filenames_u, labels_u, tfrecord_writer)


    print('\nFinished converting the data.')
