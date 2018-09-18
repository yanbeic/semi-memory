from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.slim as slim

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'padding'])
MaxPool = namedtuple('MaxPool',['kernel', 'stride'])

_CONV = [
    Conv(kernel=[3, 3], stride=1, depth=128, padding='SAME'),
    Conv(kernel=[3, 3], stride=1, depth=128, padding='SAME'),
    Conv(kernel=[3, 3], stride=1, depth=128, padding='SAME'),
    MaxPool(kernel=[2, 2], stride=[2, 2]),
    Conv(kernel=[3, 3], stride=1, depth=256, padding='SAME'),
    Conv(kernel=[3, 3], stride=1, depth=256, padding='SAME'),
    Conv(kernel=[3, 3], stride=1, depth=256, padding='SAME'),
    MaxPool(kernel=[2,2], stride=[2, 2]),
    Conv(kernel=[3, 3], stride=1, depth=512, padding='VALID'),
    Conv(kernel=[1, 1], stride=1, depth=256, padding='SAME'),
    Conv(kernel=[1, 1], stride=1, depth=128, padding='SAME'),
]


def lrelu(inputs, leak=0.1, name="lrelu"):
    with tf.name_scope(name, 'lrelu') as scope:
        return tf.maximum(inputs, leak*inputs, name=scope)


def net_base(inputs, final_endpoint='Conv2d_10', is_training=True, scope=None):
    end_points = {}
    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with slim.arg_scope([slim.conv2d]):
            net = inputs
            print('base net architectures & layer outputs')
            for i, conv_def in enumerate(_CONV):
                end_point_base = 'Conv2d_%d' % i

                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    net = slim.conv2d(inputs=net,
                                      num_outputs=conv_def.depth,
                                      kernel_size=conv_def.kernel,
                                      stride=conv_def.stride,
                                      padding=conv_def.padding,
                                      activation_fn=lrelu,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                elif isinstance(conv_def, MaxPool):
                    end_point = end_point_base + '_MaxPool'
                    net = slim.max_pool2d(inputs=net,
                                          kernel_size=conv_def.kernel,
                                          stride=conv_def.stride,
                                          padding='SAME',
                                          scope=end_point)
                    net = slim.dropout(inputs=net,
                                       keep_prob=0.5,
                                       is_training=is_training,
                                       scope=end_point + '_Dropout')
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))
                print(conv_def)
                print(net)
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def convnet_large(inputs, num_classes=1000, is_training=True, prediction_fn=tf.contrib.layers.softmax,
                  reuse=None, scope='convnet'):

  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

  with tf.variable_scope(scope, 'convnet', [inputs, num_classes], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
      net, end_points = net_base(inputs=inputs, is_training=is_training, scope=scope)
      with tf.variable_scope('Logits'):
        kernel_size = _reduced_kernel_size_for_small_input(net, [6, 6])
        net = slim.avg_pool2d(inputs=net,
                              kernel_size=kernel_size,
                              padding='VALID',
                              scope='AvgPool')
        end_points['AvgPool'] = net
        logits = slim.conv2d(inputs=net,
                             num_outputs=num_classes,
                             kernel_size=[1, 1],
                             activation_fn=None,
                             normalizer_fn=None,
                             scope='Conv2d_1c_1x1')
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      if prediction_fn:
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


convnet_large.default_image_size = 32


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def convnet_arg_scope(is_training=True, weight_decay=0.00005, stddev=0.05):
  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'decay': 0.9999,
      'epsilon': 0.001,
      'zero_debias_moving_mean': True,
  }
  weights_init = tf.random_normal_initializer(0, stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  with slim.arg_scope([slim.conv2d], weights_initializer=weights_init,
                      activation_fn=lrelu, normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer) as sc:
          return sc
