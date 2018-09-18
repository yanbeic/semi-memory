from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import nets_factory

FLAGS = tf.app.flags.FLAGS
TOWER_NAME = 'tower'
EPISILON = 1e-8


def inference(images, num_classes, for_training=False, feature_name=None):
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=num_classes,
        weight_decay=FLAGS.weight_decay,
        is_training=for_training)
    logits, end_points = network_fn(images)
    features = tf.squeeze(tf.squeeze(end_points[feature_name], squeeze_dims=1), squeeze_dims=1)
    return logits, features


def loss_ce(logits, labels, weight=1.0):
    labels = slim.one_hot_encoding(labels, FLAGS.num_classes)
    return tf.cond(tf.size(labels) > 0,
                   lambda: tf.losses.softmax_cross_entropy(
                       logits=logits,
                       onehot_labels=labels,
                       label_smoothing=FLAGS.smoothing*float(FLAGS.num_classes),
                        weights=weight),
                   lambda: tf.constant(0.0))


def loss_entropy(p_prob, weight=1.0):
    entropy = -tf.multiply(p_prob, tf.log(p_prob+EPISILON))
    return tf.multiply(weight, tf.reduce_sum(entropy, axis=1), name='entropy')


def loss_kl(p_prob, q_prob, weight=1.0):
    KL_divergence = tf.multiply(p_prob, tf.log(tf.divide(p_prob, q_prob) + EPISILON))
    return tf.multiply(weight, tf.reduce_sum(KL_divergence, axis=1), name='kl')


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        grad = tf.clip_by_value(grad, -2., 2.)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
