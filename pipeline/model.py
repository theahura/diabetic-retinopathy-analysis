import glob

import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import IPython

# Globals, hyperparameters.
FILEPATH = './model'

L2_REG = 0.005
NUM_LABELS = 5


class Model(object):

    def __init__(self):
        self.x = x = tf.placeholder(tf.float32, shape=(None, 512, 512, 3), name='x')
        self.is_training = is_training = tf.placeholder_with_default(False, shape=())
        tf.nn.leaky_relu.func_defaults = (0.3, None)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(L2_REG),
                            activation_fn=tf.nn.leaky_relu):
            # Block one.
            self.conv1 = net = slim.conv2d(x, num_outputs=32, kernel_size=[7, 7], stride=2)
            net = slim.batch_norm(net, is_training=is_training)
            self.pool1 = net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)
            # Block two.
            net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3])
            net = slim.batch_norm(net, is_training=is_training)
            self.pool2 = net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)
            # Block three.
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3])
            net = slim.batch_norm(net, is_training=is_training)
            self.pool3 = net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)
            # Block four.
            net = slim.repeat(net, 4, slim.conv2d, 128, [3, 3])
            net = slim.batch_norm(net, is_training=is_training)
            self.pool4 = net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)
            # Block five.
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3])
            net = slim.batch_norm(net, is_training=is_training)
            self.pool5 = net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

            # Dense layers.
            net = slim.flatten(net)
            self.fc1 = net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, 0.5, is_training=is_training)
            self.fc2 = net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, 0.5, is_training=is_training)
            self.logits = slim.fully_connected(net, NUM_LABELS)
            self.preds = tf.cast(tf.argmax(self.logits, 1), tf.int64)

        # Optimizer.
        self.saver = tf.train.Saver()


class Runner(object):

    def __init__(self):
        print 'init'
        self.sess = tf.Session()
        with tf.device('/cpu:0'):
            self.model = Model()
            self.model.saver = tf.train.import_meta_graph(
                glob.glob('%s/*.meta' % FILEPATH)[0], clear_devices=True)
            self.model.saver.restore(
                self.sess, tf.train.latest_checkpoint(FILEPATH))
        print 'restored'

    def get_prediction(self, fp):
        im = cv2.imread(fp)
        return self.sess.run(self.model.preds, {self.model.x: [im]})[0]


if __name__ == '__main__':
    runner = Runner()
    IPython.embed()
