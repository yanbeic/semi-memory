from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import network
import utils
from datetime import datetime

tf.app.flags.DEFINE_integer('num_classes', 100, 'The number of classes in training data.')
tf.app.flags.DEFINE_integer('num_examples', 10000, 'The number of samples in total.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'The number of samples in each batch.')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/tfmodel/', 'The directory where the model was saved.')
tf.app.flags.DEFINE_string('eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')
tf.app.flags.DEFINE_string('save_txt', None, 'The txt file to save result.')
tf.app.flags.DEFINE_string('dataset_name', 'cifar100', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string('dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('model_name', 'convnet', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string('preprocessing', 'cifar', 'The name of the preprocessing to use.')
tf.app.flags.DEFINE_float('ema_decay', 0.9999, 'If None, then not used.')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image size.')
tf.app.flags.DEFINE_string('feature_name', 'AvgPool', 'Name of the feature layer.')

FLAGS = tf.app.flags.FLAGS


def main(_):

    with tf.Graph().as_default():

        images, labels = utils.prepare_testdata(FLAGS.dataset_dir, FLAGS.batch_size)
        logits, _ = network.inference(images, FLAGS.num_classes, for_training=False, feature_name=FLAGS.feature_name)

        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_5_op = tf.nn.in_top_k(logits, labels, 5)

        var_averages = tf.train.ExponentialMovingAverage(FLAGS.ema_decay)
        var_to_restore = var_averages.variables_to_restore()
        saver = tf.train.Saver(var_to_restore)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        model_checkpoint_path = ckpt.model_checkpoint_path

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Successfully loaded model from %s at step=%s.' %
                  (model_checkpoint_path, global_step))

            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                print('num_iter = ' + str(num_iter))

                # Counts the number of correct predictions.
                count_top_1 = count_top_5 = 0.0
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0

                print('%s: starting evaluation on (%s).' % (datetime.now(), 'test'))
                start_time = time.time()
                while step < num_iter and not coord.should_stop():
                    top_1, top_5 = sess.run([top_1_op, top_5_op])
                    count_top_1 += np.sum(top_1)
                    count_top_5 += np.sum(top_5)
                    step += 1
                    # print progress every 20 batches
                    if step % 20 == 0:
                        duration = time.time() - start_time
                        sec_per_batch = duration / 20.0
                        examples_per_sec = FLAGS.batch_size / sec_per_batch
                        print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f sec/batch)'
                              % (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                        start_time = time.time()

                # Compute precision @ 1. (accuracy) and print results
                precision_at_1 = count_top_1 / total_sample_count
                recall_at_5 = count_top_5 / total_sample_count
                print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
                      (datetime.now(), precision_at_1, recall_at_5, total_sample_count))

                # save results into a txt file
                file_path = FLAGS.eval_dir+FLAGS.save_txt
                text_file = open(file_path, 'a')
                text_file.write(FLAGS.checkpoint_path)
                text_file.write('\n')
                text_file.write('%s: precision @ 1 = %.4f recall @ 5 = %.4f' %
                                (datetime.now(), precision_at_1, recall_at_5))
                text_file.write('\n')
                text_file.close()

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':
    tf.app.run()
