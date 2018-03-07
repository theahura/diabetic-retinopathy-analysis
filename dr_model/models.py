"""
Tensorflow models.
"""

import tensorflow.contrib.slim as slim
import IPython

import inception_resnet as ir


def basic_model(x, activation, l2_reg, is_training):

    endpoints = {}

    with slim.arg_scope([slim.conv2d],
                        activation_fn=activation,
                        weights_regularizer=slim.l2_regularizer(l2_reg)):
        # Block one.
        net = slim.conv2d(x, num_outputs=32, kernel_size=7, stride=2)
        endpoints['conv1'] = net
        net = slim.max_pool2d(net, kernel_size=3)
        endpoints['pool1'] = net

        # Block two.
        net = slim.repeat(net, 2, slim.conv2d, 32, 3)
        net = slim.batch_norm(net, is_training=is_training)
        net = slim.max_pool2d(net, kernel_size=3)
        endpoints['pool2'] = net

        # Block three.
        net = slim.repeat(net, 2, slim.conv2d, 64, 3)
        net = slim.batch_norm(net, is_training=is_training)
        net = slim.max_pool2d(net, kernel_size=3)
        endpoints['pool3'] = net

        # Block four.
        net = slim.repeat(net, 4, slim.conv2d, 128, 3)
        net = slim.batch_norm(net, is_training=is_training)
        net = slim.max_pool2d(net, kernel_size=3)
        endpoints['pool4'] = net

    return net, endpoints


def inception_resnet(x, is_training, activation):
    scope = ir.inception_resnet_v2_arg_scope(activation_fn=activation,
                                             is_training=is_training)
    with slim.arg_scope(scope):
        return ir.inception_resnet_v2_base(x, activation_fn=activation)
