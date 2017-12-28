"""
Batcher, model, training.
"""

import os

import cv2
import numpy as np
import pandas
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading

import IPython

rng = np.random.RandomState(128)

# Globals, hyperparameters.
SUMMARIES_EVERY_N = 5
VALIDATION_EVERY_N = 10
BATCH_SIZE = 16
VAL_BATCH_SIZE = 16

L2_REG = 0.0005
LEARN_RATE = 0.1

RESTORE = True
LAST_N_VAL = 10
NUM_LABELS = 5
LABELS_FP = './trainLabels.csv'
DATA_FP = './train'
LOGDIR = './logs/'
CKPTS = './ckpts/'


def get_accuracies(session, preds, labels):
    accs = np.zeros(NUM_LABELS)
    counts = np.bincount(labels, minlength=NUM_LABELS)
    for p, l in zip(preds, labels):
        if p == l:
            accs[l] += 1
    accs /= counts

    print "Accuracies:"
    print accs
    print "Counts:"
    print counts
    print "Predictions:"
    print preds

    return accs


def get_accuracy_summaries(session, accs, op, plhlds):
    inputs = {plhlds[i]: accs[i] for i in range(NUM_LABELS)}
    return session.run(op, inputs)


def open_files(f):
    return cv2.imread(f)


class Batcher(object):

    def __init__(self, data_fp, label_fp, validation=False):

        # Set initial vars.
        self.index = 0
        self.val_flag = False
        self.next_batch_flag = False
        self.next_batch_data = None
        self.next_batch_arrays = None
        self.val_arrays = None

        # Get initial data.
        files = [os.path.join(data_fp, f) for f in os.listdir(data_fp)]

        self.data = pandas.read_csv(label_fp)

        # Add filepath.
        files = sorted(files, key=lambda f: int(f.split('/')[-1].split('_')[0]))
        self.data['filename'] = files

        # Balance class weights: num_samples/(num_classes*num_obs_per_class).
        counts = self.data['level'].value_counts()
        c = len(counts)
        n = max(counts)
        weights = [n/(c * counts[level] * 1.0) for level in self.data['level']]
        weights = np.square(np.array(weights))
        self.data['weights'] = weights

        # Get onehots.
        onehots = np.eye(NUM_LABELS, dtype=int)
        def to_onehot(i):
            return onehots[i]

        self.data['labels'] = self.data['level'].map(to_onehot)

        # Set the validation set to the last fifth of the training data.
        if validation:
            val_slice = len(self.data)/5
            self.val_data = self.data.iloc[-val_slice:]
            self.data = self.data.iloc[:-val_slice]
            print 'Getting validation arrays'
            thread = threading.Thread(target=self._load_val_arrays)
            thread.daemon = True
            thread.start()
        thread = threading.Thread(target=self._get_batch, args=(BATCH_SIZE,))
        thread.daemon = True
        thread.start()

    def _load_val_arrays(self):
        self.val_arrays = np.stack(self.val_data['filename'].map(open_files).values)
        self.val_flag = True

    def _get_batch(self, size, step=0):
        #batch_data = self.data.iloc[self.index:self.index+size]
        current_level = step % NUM_LABELS
        batch_data = self.data.loc[self.data['level'] == current_level].sample(size)
        #self.index += size

        #if self.index > len(self.data):
        #    self.index = 0

        batch_arrays = np.stack(batch_data['filename'].map(open_files).values)
        self.next_batch_data = batch_data
        self.next_batch_arrays = batch_arrays
        self.next_batch_flag = True

    def get_batch(self, size, step):
        while not self.next_batch_flag: pass
        next_batch = self.next_batch_data, self.next_batch_arrays
        self.next_batch_flag = False
        thread = threading.Thread(target=self._get_batch, args=(size,step))
        thread.daemon = True
        thread.start()
        return next_batch

    def get_validation_batch(self, size):
        while not self.val_flag: pass
        indices = np.random.choice(len(self.val_data), size, replace=False)
        return self.val_data.iloc[indices], self.val_arrays[indices]

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
            #net = slim.dropout(net, 0.5, is_training=is_training)
            net = slim.fully_connected(net, 1024)
            #net = slim.dropout(net, 0.5, is_training=is_training)
            self.logits = slim.fully_connected(net, NUM_LABELS)
            self.preds = tf.argmax(self.logits, 1)

        # Losses.
        # Dont use weights if using skewed batch sampling.
        self.labels = tf.placeholder(tf.float32, name='labels')
        self.crossent_loss = tf.losses.softmax_cross_entropy(self.labels,
                                                             self.logits)
                                                             #self.weights)
        self.norm_loss = tf.losses.get_regularization_losses()
        self.total_loss = tf.losses.get_total_loss()

        # Optimizer.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #op = tf.train.AdamOptimizer(LEARN_RATE)
        op = tf.train.GradientDescentOptimizer(0.001)
        net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.grads = tf.gradients(self.total_loss, net_vars)
        grads_and_vars = list(zip(self.grads, net_vars))
        self.train = op.apply_gradients(grads_and_vars,
                                        global_step=self.global_step)

        # Model Summaries.
        _loss = tf.summary.scalar('model/tot_loss', tf.reduce_mean(self.total_loss))
        _reg = tf.summary.scalar('model/reg_losses', tf.reduce_mean(self.norm_loss))
        _grad_norm = tf.summary.scalar('model/grad_norm', tf.global_norm(self.grads))
        _var_norm = tf.summary.scalar('model/var_norm', tf.global_norm(net_vars))
        _logits = tf.summary.histogram('model/preds', self.preds)

        self.summary_op = tf.summary.merge([_loss, _reg, _grad_norm, _var_norm,
                                            _logits])

        self.saver = tf.train.Saver()


# Init batcher, model, session.
batcher = Batcher(DATA_FP, LABELS_FP, True)
model = Model()
sess = tf.Session()

# Summary writer, acc ops.
writer = tf.summary.FileWriter(LOGDIR, sess.graph, flush_secs=60)

_acc_plhlds = [tf.placeholder(tf.float32, name='acc_' + str(i)) for i in range(NUM_LABELS)]
_accs = [tf.summary.scalar('acc/' + str(i), _acc_plhlds[i]) for i in range(NUM_LABELS)]
_acc_sums = tf.summary.merge(_accs)

# Training.
step = 0
sess.run(tf.global_variables_initializer())
if RESTORE:
    print 'RESTORING'
    ckpt = tf.train.get_checkpoint_state(CKPTS)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print 'restore failed'

IPython.embed()

try:
    for i in range(0, 1000):
        if step % VALIDATION_EVERY_N == 0 and step > 1:
            print 'Validation'
            model.saver.save(sess, CKPTS, global_step=sess.run(model.global_step))
            batch_data, batch_arrays = batcher.get_validation_batch(VAL_BATCH_SIZE)
            inputs = {model.x: batch_arrays}
            preds = sess.run(model.preds, inputs)
            accs = get_accuracies(sess, preds, batch_data['level'].as_matrix())
            accuracy_sum = get_accuracy_summaries(sess, accs, _acc_sums, _acc_plhlds)

            writer.add_summary(tf.Summary.FromString(accuracy_sum),
                               sess.run(model.global_step))

            writer.flush()
        else:
            batch_data, batch_arrays = batcher.get_batch(BATCH_SIZE, step)
            print "Got batch"
            print batch_data['level'].values
            inputs = {
                model.x: batch_arrays,
                model.weights: batch_data['weights'].as_matrix(),
                model.labels: np.stack(batch_data['labels'].values)
            }

            if step % SUMMARIES_EVERY_N == 0 and step > 1:
                print 'Summary'
                fetched = sess.run([model.train, model.summary_op], inputs)
                writer.add_summary(tf.Summary.FromString(fetched[1]),
                                   sess.run(model.global_step))
                writer.flush()
            else:
                print 'Training'
                fetched = sess.run([model.train, model.total_loss], inputs)
                print 'Loss: %f' % fetched[1]

        step += 1
        print i
except KeyboardInterrupt:
   IPython.embed()
