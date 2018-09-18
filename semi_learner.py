from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import network
import memory
import utils
import math

FLAGS = tf.app.flags.FLAGS


def _build_training_graph(images, labels, num_classes, reuse_variables=None):

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        logits, features = \
            network.inference(images, num_classes, for_training=True, feature_name=FLAGS.feature_name)

    labels_l, logits_l, logits_u, features_l, features_u = memory.label_ulabel(labels, logits, features)

    keys, values = memory.module(reuse_variables, labels_l, features_l, tf.nn.softmax(logits_l))

    mem_pred, net_pred = memory.assimilation(keys, values, labels_l, features_u, logits)

    loss_s = network.loss_ce(logits_l, labels_l)
    loss_m = memory.accomodation(mem_pred, net_pred)*FLAGS.weight_u
    losses = [loss_s + loss_m]

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % network.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name + ' (raw)', l)
        tf.summary.scalar(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

    return total_loss, loss_s, loss_m, labels_l, logits_l, [keys, values]


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        assert FLAGS.batch_size % FLAGS.num_gpus == 0, ('Batch size must be divisible by number of GPUs')

        bs_l = FLAGS.batch_size * FLAGS.label_ratio
        bs_u = FLAGS.batch_size * (1 - FLAGS.label_ratio)
        num_iter_per_epoch = int(FLAGS.num_train_u / bs_u)
        max_steps = int(FLAGS.num_epochs * num_iter_per_epoch)
        num_classes = FLAGS.num_classes

        global_step = slim.create_global_step()
        lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)

        images_l, labels_l = utils.prepare_traindata(FLAGS.dataset_dir_l, int(bs_l))
        images_u, labels_u = utils.prepare_traindata(FLAGS.dataset_dir_u, int(bs_u))

        images_splits_l = tf.split(images_l, FLAGS.num_gpus, 0)
        images_splits_u = tf.split(images_u, FLAGS.num_gpus, 0)
        labels_splits_l = tf.split(labels_l, FLAGS.num_gpus, 0)
        labels_splits_u = tf.split(labels_u, FLAGS.num_gpus, 0)

        images_splits = []
        labels_splits = []
        for i in range(FLAGS.num_gpus):
            images_splits.append(tf.concat([images_splits_l[i], images_splits_u[i]], 0))
            labels_splits.append(tf.concat([labels_splits_l[i], labels_splits_u[i]], 0))

        tower_grads = []
        top_1_op = []
        memory_op = []
        reuse_variables = None
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (network.TOWER_NAME, i)) as scope:
                    with slim.arg_scope(slim.get_model_variables(scope=scope), device='/cpu:0'):
                        loss, loss_s, loss_m, labels, logits, memory_update = \
                            _build_training_graph(images_splits[i], labels_splits[i], num_classes, reuse_variables)

                        memory_op.append(memory_update)
                        top_1_op.append(tf.nn.in_top_k(logits, labels, 1))

                    reuse_variables = True
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    batchnorm = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = network.average_gradients(tower_grads)
        gradient_op = opt.apply_gradients(grads, global_step=global_step)

        var_averages = tf.train.ExponentialMovingAverage(FLAGS.ema_decay, global_step)
        var_op = var_averages.apply(tf.trainable_variables())

        batchnorm_op = tf.group(*batchnorm)
        train_op = tf.group(gradient_op, var_op, batchnorm_op)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        summary_op = tf.summary.merge(summaries)
        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        if FLAGS.gpu_memory:
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory
        sess = tf.Session(config=config)

        boundaries, values = utils.config_lr(max_steps)
        sess.run([init_op], feed_dict={lr: values[0]})

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

        iter_count = epoch = sum_loss = sum_loss_s = sum_loss_m = sum_top_1 = 0
        start = time.time()

        for step in range(max_steps):

            decayed_lr = utils.decay_lr(step, boundaries, values, max_steps)
            _, _, loss_value, loss_value_s, loss_value_m, top_1_value = \
                sess.run([train_op, memory_op, loss, loss_s, loss_m, top_1_op], feed_dict={lr: decayed_lr})

            sum_loss += loss_value
            sum_loss_s += loss_value_s
            sum_loss_m += loss_value_m
            top_1_value = np.sum(top_1_value) / bs_l
            sum_top_1 += top_1_value
            iter_count +=1

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            assert not np.isnan(loss_value_s), 'Model diverged with loss = NaN'
            assert not np.isnan(loss_value_m), 'Model diverged with loss = NaN'

            if step % num_iter_per_epoch == 0 and step > 0:
                end = time.time()
                sum_loss = sum_loss / num_iter_per_epoch
                sum_loss_s = sum_loss_s / num_iter_per_epoch
                sum_loss_m = sum_loss_m / num_iter_per_epoch
                sum_top_1 = min(sum_top_1 / num_iter_per_epoch, 1.0)
                time_per_iter = float(end - start) / iter_count
                format_str = ('epoch %d, L = %.2f, Ls = %.2f, Lm = %.2f, top_1 = %.2f, lr = %.4f (time_per_iter: %.4f s)')
                print(format_str % (epoch, sum_loss, sum_loss_s, sum_loss_m, sum_top_1*100, decayed_lr, time_per_iter))
                epoch +=1
                sum_loss = sum_loss_s = sum_loss_m = sum_top_1 = 0

            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={lr: decayed_lr})
                summary_writer.add_summary(summary_str, step)

            if (step + 1) == max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
