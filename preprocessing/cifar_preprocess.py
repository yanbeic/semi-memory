from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from preprocessing import utils

_PADDING = 2
_IMAGE_SIZE = 32


def preprocess_for_train(image, height=_IMAGE_SIZE, width=_IMAGE_SIZE, padding=_PADDING):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = utils.flip(image)
    image = utils.translation(image, height, width, padding)
    image = utils.distort_color(image)
    image = utils.clip(image)
    return image


def preprocess_for_eval(image, height=_IMAGE_SIZE, width=_IMAGE_SIZE):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    return image


def preprocess_image(image, height, width, is_training=False):
    if is_training:
        return preprocess_for_train(image, height, width)
    else:
        return preprocess_for_eval(image, height, width)
