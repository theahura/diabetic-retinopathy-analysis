import os

import numpy as np
import pandas
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim

import IPython

rng = np.random.RandomState(128)

# Globals, hyperparameters.
PREPROCESS_DATA_PARAMS = 100

NUM_LABELS = 5
L2_REG = 0.0005
LEARN_RATE = 0.1

LABELS_FP = './trainLabels.csv'
DATA_FP = './data'



def get_dataset_params(data):
    IPython.embed()

class Batcher(object):

    def __init__(self):

        # Set index.
        self.index = 0

        # Get initial data.
        files = [os.path.join(DATA_FP, f) for f in os.listdir(DATA_FP)]
        self.data = pandas.read_csv(LABELS_FP)

        # Add filepath.
        files = sorted(files, key=lambda f: int(f.split('/')[-1].split('_')[0]))
        self.data['filename'] = files

        # Balance class weights: num_samples/(num_classes*num_obs_per_class).
        counts = self.data['level'].value_counts()
        n = len(self.data)
        c = len(counts)
        weights = [n/(c * counts[level] * 1.0) for level in self.data['level']]
        self.data['weights'] = weights

        # Get data parameters.
        self.parameters = get_dataset_params(self.data)

    def get_batch(self, size):
        batch_data = self.data[self.index:self.index+size]
        pass

# Create network.
class Model(object):

    def __init__(self):
        self.x = x = tf.placeholder(tf.float32, shape=(None, 512, 512, 3), name='x')
        self.is_training = is_training = tf.placeholder_with_default(True, shape=())
        self.weights = tf.placeholder(tf.float32, name='weights')
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(L2_REG)):
            # Block one.
            net = slim.conv2d(x, num_outputs=32, kernel_size=[7, 7], stride=2)
            net = slim.batch_norm(net, is_training=is_training)
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2)
            # Block two.
            net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3])
            net = slim.batch_norm(net, is_training=is_training)
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2)
            # Block three.
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3])
            net = slim.batch_norm(net, is_training=is_training)
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2)
            # Block four.
            net = slim.repeat(net, 4, slim.conv2d, 128, [3, 3])
            net = slim.batch_norm(net, is_training=is_training)
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2)
            # Block five.
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3])
            net = slim.batch_norm(net, is_training=is_training)
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2)

            # Dense layers.
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, 0.5, is_training=is_training)
            net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, 0.5, is_training=is_training)
            self.logits = slim.fully_connected(net, NUM_LABELS)

        # Losses.
        self.labels = tf.placeholder(tf.float32, shape=(None, NUM_LABELS))
        self.crossent_loss = tf.losses.softmax_cross_entropy(self.logits,
                                                             self.labels,
                                                             self.weights)
        self.norm_loss = tf.losses.get_regularization_losses()
        self.total_loss = tf.losses.get_total_loss()

        # Optimizer.
        op = tf.train.AdamOptimizer(LEARN_RATE)
        net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.grads = tf.gradients(self.total_loss, net_vars)
        grads_and_vars = list(zip(self.grads, net_vars))
        self.train = op.apply_gradients(grads_and_vars)

batcher = Batcher()
model = Model()

IPython.embed()
