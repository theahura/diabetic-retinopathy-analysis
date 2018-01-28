import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim
import IPython

import data_preprocess as dp

# Globals, hyperparameters.
FILEPATH = './model'

L2_REG = 0.005
NUM_LABELS = 5
LEAKY = 0.5


# Create network.
class Model(object):

    def __init__(self):
        self.x = x = tf.placeholder(tf.float32, shape=(None, 512, 512, 3), name='x')
        self.is_training = is_training = tf.placeholder_with_default(True, shape=())

        def leaky(x):
            return tf.nn.leaky_relu(x, LEAKY)

        with slim.arg_scope([slim.conv2d],
                            activation_fn=leaky,
                            weights_regularizer=slim.l2_regularizer(L2_REG)):
            # Block one.
            self.conv1 = net = slim.conv2d(x, num_outputs=32, kernel_size=7,
                                           stride=2)
            self.pool1 = net = slim.max_pool2d(net, kernel_size=3)
            # Block two.
            net = slim.repeat(net, 2, slim.conv2d, 32, 3)
            net = slim.batch_norm(net, is_training=self.is_training)
            self.pool2 = net = slim.max_pool2d(net, kernel_size=3)
            # Block three.
            net = slim.repeat(net, 2, slim.conv2d, 64, 3)
            net = slim.batch_norm(net, is_training=self.is_training)
            self.pool3 = net = slim.max_pool2d(net, kernel_size=3)
            # Block four.
            net = slim.repeat(net, 4, slim.conv2d, 128, 3)
            net = slim.batch_norm(net, is_training=self.is_training)
            self.pool4 = net = slim.max_pool2d(net, kernel_size=3)

            # Block five; GAP.
            self.feats = slim.conv2d(net, num_outputs=1024, kernel_size=3)
            self.gap = tf.reduce_mean(self.feats, [1, 2])
            net = slim.dropout(self.gap, 0.5, is_training=is_training)

        self.logits = slim.fully_connected(net, NUM_LABELS,
                                           activation_fn=None,
                                           weights_regularizer=slim.l2_regularizer(L2_REG),
                                           scope='fc')
        sftmx = tf.nn.softmax(self.logits)
        self.preds = tf.cast(tf.argmax(sftmx, 1), tf.int64)

        self.saver = tf.train.Saver()

        def heatmap(index):
            pred = self.preds[index]
            fcs = [l for l in tf.global_variables() if l.name.startswith('fc')]
            weights = tf.expand_dims(tf.transpose(fcs[0])[pred], axis=1)
            f = self.feats[index]
            h, w, c = f.shape
            cam = tf.matmul(tf.reshape(f, (h*w, c)), weights)
            cam = tf.reshape(cam, (h, w, 1))
            cam = tf.subtract(cam, tf.reduce_min(cam))
            cam = tf.div(cam, tf.reduce_max(cam))
            cam = tf.expand_dims(cam, axis=0)
            cam = tf.image.resize_bilinear(cam, [512, 512])
            return cam 

        self.heatmap = heatmap(0)


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
        im = dp.process_image(fp)
        outputs = [self.model.preds, self.model.heatmap]
        preds, hms = self.sess.run(outputs, {self.model.x: [im]})
        return im, preds, hms


if __name__ == '__main__':
    runner = Runner()
