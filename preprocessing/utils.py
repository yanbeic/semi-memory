from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

brightness = lambda x: tf.image.random_brightness(x, max_delta=63. / 255.)
saturation = lambda x: tf.image.random_saturation(x, lower=0.2, upper=1.8)
contrast = lambda x: tf.image.random_contrast(x, lower=0.2, upper=1.8)


def random_num(num):
    select_num = tf.random_uniform(shape=[], minval=1, maxval=num + 1, dtype=tf.int32)
    return select_num


def pad(image, padding):
    image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]], mode='REFLECT')
    return image


def set_shape(image, h, w):
    def fn1(): return tf.random_crop(image, [h, w, 3])
    def fn2(): return tf.image.resize_images(image, [h, w])
    image = tf.cond(tf.equal(random_num(2), 1), fn1, fn2)
    image.set_shape([h, w, 3])
    return image


def translation(image, h, w, padding):
    image = pad(image, padding)
    image = set_shape(image, h, w)
    return image


def flip(image):
    image = tf.image.random_flip_left_right(image)
    return image


def rotation(image):
    degree = tf.random_uniform(shape=[1], minval=-0.2, maxval=0.2)
    image = tf.contrib.image.rotate(image, degree, interpolation='BILINEAR')
    return image


def distort_color(image):
    def fn1(): return contrast(saturation(brightness(image)))
    def fn2(): return saturation(contrast(brightness(image)))
    def fn3(): return contrast(brightness(saturation(image)))
    def fn4(): return brightness(contrast(saturation(image)))
    def fn5(): return saturation(brightness(contrast(image)))
    def fn6(): return brightness(saturation(contrast(image)))
    def fn(): return image
    cond = random_num(6)
    image = tf.cond(tf.equal(cond, 1), fn1, fn)
    image = tf.cond(tf.equal(cond, 2), fn2, fn)
    image = tf.cond(tf.equal(cond, 3), fn3, fn)
    image = tf.cond(tf.equal(cond, 4), fn4, fn)
    image = tf.cond(tf.equal(cond, 5), fn5, fn)
    image = tf.cond(tf.equal(cond, 6), fn6, fn)
    return image


def clip(image):
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image
