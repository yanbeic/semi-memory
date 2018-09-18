from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import network

FLAGS = tf.app.flags.FLAGS
EPISILON=1e-10

mask = lambda x, y: tf.boolean_mask(x, y)
diff = lambda x, n, eta: (x / tf.cast((1 + n), tf.float32))*eta
normalize = lambda x: x / tf.reduce_sum(x, axis=1, keep_dims=True)


def module(reuse_variables, labels, features, inferences):
    num_c = FLAGS.num_classes
    dim_f = FLAGS.feature_dim
    with tf.variable_scope("memory", reuse=reuse_variables):
        keys = tf.get_variable('keys',
                               shape=[num_c, dim_f],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0),
                               trainable=False)
        values = tf.get_variable('values',
                                 shape=[num_c, num_c],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1.0 / float(num_c)),
                                 trainable=False)

    diff_key = tf.gather(keys, labels) - features
    diff_value = tf.gather(values, labels) - inferences

    y, idx, count = tf.unique_with_counts(labels)
    count_n = tf.expand_dims(tf.gather(count, idx), 1)

    diff_key = diff(diff_key, count_n, FLAGS.eta)
    diff_value = diff(diff_value, count_n, FLAGS.eta)

    keys = tf.scatter_sub(keys, labels, diff_key)
    values = normalize(tf.scatter_sub(values, labels, diff_value))

    return keys, values


def label_ulabel(labels, logits, features):
    where_label = tf.not_equal(labels, -1) # unlabel is given as -1
    where_unlabel = tf.equal(labels, -1)
    labels_l = mask(labels, where_label)
    logits_l = mask(logits, where_label)
    logits_u = mask(logits, where_unlabel)
    features_l = mask(features, where_label)
    features_u = mask(features, where_unlabel)
    return labels_l, logits_l, logits_u, features_l, features_u


def content_based(keys, values, features_u):
    dist = tf.sqrt((features_u[:, tf.newaxis] - keys) ** 2 + EPISILON)
    memberships = tf.nn.softmax(-tf.reduce_sum(dist, axis=2))
    memberships = tf.clip_by_value(memberships, EPISILON, 1)
    pred_u = normalize(tf.reduce_sum(memberships[:, tf.newaxis] * values, 2))
    return pred_u


def position_based(values, labels_l):
    pred_l = tf.gather(values, labels_l)
    return pred_l


def memory_prediction(keys, values, labels_l, features_u):
    # key addressing & value reading
    pred_l = position_based(values, labels_l)
    pred_u = content_based(keys, values, features_u)
    return tf.concat([pred_l, pred_u], 0, name='memory_pred')


def assimilation(keys, values, labels_l, features_u, logits):
    mem_pred = memory_prediction(keys, values, labels_l, features_u)
    net_pred = tf.nn.softmax(logits)
    return mem_pred, net_pred


def accomodation(mem_pred, net_pred):
   # model entropy
   m_entropy = network.loss_entropy(mem_pred)
   # memory-network divergence
   mn_divergence = network.loss_kl(net_pred, mem_pred)
   uncertainty = tf.reduce_max(mem_pred, axis=1)
   mnd_weighted = tf.multiply(mn_divergence, uncertainty)
   loss_m = tf.reduce_mean(tf.add(m_entropy, mnd_weighted))
   return loss_m
