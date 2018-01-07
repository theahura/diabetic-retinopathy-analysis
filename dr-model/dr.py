"""
Batcher, model, training.

TODO:
    - add precision and recall
    - fix randomness in batching to avoid random oversampling
    - play with dense layers
    - normalize (sub mean, divide std div) per batch/across dataset
    - consider removing grad norm summary
    - ensure create_train_op is used correctly
    - check memory errors
"""

import os
import random

import cv2
import numpy as np
import pandas
from PIL import Image, ImageChops, ImageOps, ImageEnhance
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading

import IPython

rng = np.random.RandomState(128)
random.seed(a=128)
tf.set_random_seed(128)

# Globals, hyperparameters.
NUM_STEPS = 80000
EVEN_CUTOFF = NUM_STEPS/100 * 60
SUMMARIES_EVERY_N = 20
VALIDATION_EVERY_N = 40

L2_REG = 0.0005
LEAKY = 0.5
MOMENTUM = 0.9
GRAD_NORM = 1000
LR_SCALE = 6.00
BOUNDARIES = [NUM_STEPS / 100 * i for i in [30, 50, 85, 95]]
VALUES = [i * LR_SCALE for i in [0.001, 0.0005, 0.0001, 0.00001, 0.000001]]

NUM_BATCH_THREADS = 5
LOADED_BATCH = 7
BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
TRANSLATE = False
ROTATE = True
FLIP = True

LABELS_FP = './trainLabels.csv'
DATA_FP = './train_2'
CKPTS_RESTORE = './ckpts'
LOGDIR = './logs/'

RESTORE = True
NUM_LABELS = 5

fname = 'batchsize-%d_l2-%f_lr-%f-train-%s-%f_cutoff-%d_leaky-%f' % (
    BATCH_SIZE, L2_REG, LR_SCALE, 'nesterov', MOMENTUM, EVEN_CUTOFF, LEAKY)

CKPTS_SAVE = CKPTS_RESTORE + '/' + fname


def get_accuracies(session, preds, labels):
    accs = np.zeros(NUM_LABELS + 1)
    counts = np.append(np.bincount(labels, minlength=NUM_LABELS), len(labels))
    for p, l in zip(preds, labels):
        if p == l:
            accs[l] += 1
            accs[-1] += 1
    accs /= counts

    print "Accuracies:"
    print accs
    print "Counts:"
    print counts
    print "Labels:"
    print labels
    print "Predictions:"
    print preds

    return accs


def get_accuracy_summaries(session, accs, op, plhlds):
    inputs = {plhlds[i]: accs[i] for i in range(len(accs))}
    return session.run(op, inputs)


def open_files(f):
    im = Image.open(f)

    rows, cols = im.size

    # contrast = np.random.uniform(0.7, 1.3)
    # brightness = np.random.uniform(0.7, 1.3)
    # color = np.random.uniform(0.7, 1.3)
    # im = ImageEnhance.Contrast(im).enhance(contrast)
    # im = ImageEnhance.Brightness(im).enhance(brightness)
    # im = ImageEnhance.Color(im).enhance(color)

    # Rotation.
    if ROTATE:
        im = im.rotate(random.randint(0, 359))

    # Translation.
    if TRANSLATE:
        y_trans = random.randint(-rows/6, rows/6)
        x_trans = random.randint(-cols/6, cols/6)
        im = ImageChops.offset(im, x_trans, y_trans)

    # Flip.
    if FLIP:
        choice = random.randint(0, 1)
        if choice == 0:
            im = ImageOps.flip(im)

    im = np.asarray(im, dtype=np.uint8)
    im.setflags(write=1)
    im[np.where((im == [0, 0, 0]).all(axis=2))] = [128, 128, 128]
    return im


def start_daemon_thread(target, args):
    thread = threading.Thread(target=target, args=args)
    thread.daemon = True
    thread.start()


class Batcher(object):
    """
    Handles minibatching.
    Spins off threads that preload batches in arrays, which are then retrieved
    by training loop.
    """

    def __init__(self, data_fp, label_fp, validation=False):

        # Set initial vars.
        self.index = 0
        self.even_batch = True
        self.next_batch_data = []
        self.next_batch_arrays = []
        self.next_val_data = []
        self.next_val_arrays = []

        # Get initial data.
        files = [os.path.join(data_fp, f) for f in os.listdir(data_fp)]

        self.data = pandas.read_csv(label_fp).sample(frac=1, random_state=rng)

        # Add filepath.
        files = sorted(files, key=lambda f: int(f.split('/')[-1].split('_')[0]))
        self.data['filename'] = files

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
            start_daemon_thread(self._get_val_batch, (VAL_BATCH_SIZE,))

        for _ in range(NUM_BATCH_THREADS):
            start_daemon_thread(self._get_batch, args=(BATCH_SIZE,))

    def _get_val_batch(self, size):
        """Preloads two validation minibatches."""
        while True:
            if len(self.next_val_data) >= 2 and len(self.next_val_arrays) >= 2:
                continue
            indices = np.random.choice(len(self.val_data), size, replace=False)
            batch_data = self.val_data.iloc[indices]
            batch_arrays = np.stack(batch_data['filename'].map(open_files).values)

            self.next_val_data.append(batch_data)
            self.next_val_arrays.append(batch_arrays/255.0)

    def _get_distribution_batch(self, size):
        """Samples a minibatch according to the true distribution."""
        batch_data = self.data.iloc[self.index:self.index + size]
        self.index += size

        if self.index > len(self.data):
            self.index = 0

        batch_arrays = np.stack(batch_data['filename'].map(open_files).values)
        return batch_data, batch_arrays/255.0

    def _get_even_batch(self, size):
        """Samples a minibatch evenly across classes."""
        batch_data = pandas.DataFrame()
        for i in range(NUM_LABELS):
            label_data = self.data.loc[
                self.data['level'] == i].sample(size/NUM_LABELS, random_state=rng)
            batch_data = batch_data.append(label_data)

        if len(batch_data) < size:
            rem = size - len(batch_data)
            rem_class = random.randint(0, NUM_LABELS - 1)
            label_data = self.data.loc[
                self.data['level'] == rem_class].sample(rem, random_state=rng)
            batch_data = batch_data.append(label_data)

        batch_arrays = np.stack(batch_data['filename'].map(open_files).values)
        return batch_data, batch_arrays/255.0

    def _get_batch(self, size):
        """Preloads five train minibatches."""
        while True:
            if (len(self.next_batch_data) >= LOADED_BATCH and
                len(self.next_batch_arrays) >= LOADED_BATCH):
                continue
            if self.even_batch:
                data, arrays = self._get_even_batch(size)
            else:
                data, arrays = self._get_distribution_batch(size)
            self.next_batch_data.append(data)
            self.next_batch_arrays.append(arrays)

    def get_batch(self):
        while not (len(self.next_batch_data) and len(self.next_batch_arrays)):
            pass
        return self.next_batch_data.pop(0), self.next_batch_arrays.pop(0)

    def get_validation_batch(self):
        while not (len(self.next_val_data) and len(self.next_val_arrays)):
            pass
        return self.next_val_data.pop(0), self.next_val_arrays.pop(0)


# Create network.
class Model(object):

    def __init__(self):
        self.x = x = tf.placeholder(tf.float32, shape=(None, 512, 512, 3), name='x')
        self.is_training = is_training = tf.placeholder_with_default(True, shape=())
        tf.nn.leaky_relu.func_defaults = (LEAKY, None)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(L2_REG),
                            activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
            # Block one.
            self.conv1 = net = slim.conv2d(x, num_outputs=32, kernel_size=7,
                                           stride=2)
            net = slim.batch_norm(net, is_training=is_training)
            self.pool1 = net = slim.max_pool2d(net, kernel_size=3)
            # Block two.
            net = slim.repeat(net, 2, slim.conv2d, 32, 3)
            self.pool2 = net = slim.max_pool2d(net, kernel_size=3)
            # Block three.
            net = slim.repeat(net, 2, slim.conv2d, 64, 3)
            self.pool3 = net = slim.max_pool2d(net, kernel_size=3)
            # Block four.
            net = slim.repeat(net, 4, slim.conv2d, 128, 3)
            self.pool4 = net = slim.max_pool2d(net, kernel_size=3)
            # Block five.
            net = slim.repeat(net, 4, slim.conv2d, 256, 3)
            self.pool5 = net = slim.max_pool2d(net, kernel_size=3)

            # Dense layers.
            self.flattened = net = slim.flatten(net)
            net = slim.dropout(net, 0.5, is_training=is_training)
            self.fc1 = net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, 0.5, is_training=is_training)
            self.fc1 = net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, 0.5, is_training=is_training)
            self.logits = slim.fully_connected(net, NUM_LABELS)
            self.preds = tf.cast(tf.argmax(self.logits, 1), tf.int64)

        # Losses.
        self.labels = tf.placeholder(tf.int32, name='labels')
        self.crossent_loss = tf.losses.softmax_cross_entropy(self.labels,
                                                             self.logits)
        self.norm_loss = tf.losses.get_regularization_losses()
        self.total_loss = tf.losses.get_total_loss()

        # Optimizer.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(self.global_step,
                                                    BOUNDARIES, VALUES)
        op = tf.train.MomentumOptimizer(learning_rate, MOMENTUM, use_nesterov=True)
        #op = tf.train.AdamOptimizer(LEARN_RATE)
        #op = tf.train.GradientDescentOptimizer(0.001)

        self.train = slim.learning.create_train_op(self.total_loss, op,
                                                   clip_gradient_norm=GRAD_NORM)

        # Model Summaries.
        _loss = tf.summary.scalar('model/tot_loss', tf.reduce_mean(self.total_loss))
        _reg = tf.summary.scalar('model/reg_losses', tf.reduce_mean(self.norm_loss))
        tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(self.total_loss, tvs)
        _grad_norm = tf.summary.scalar('model/grad_norm', tf.global_norm(grads))
        _var_norm = tf.summary.scalar('model/var_norm', tf.global_norm(tvs))
        _logits = tf.summary.histogram('model/preds', self.preds)
        levels = tf.argmax(self.labels, 1)
        _t_acc = tf.summary.scalar(
            'acc/train_acc', tf.contrib.metrics.accuracy(self.preds, levels))

        def reshape_filters(layer, label=0):
            label_index = label * (BATCH_SIZE / NUM_LABELS)
            return tf.expand_dims(tf.transpose(layer[label_index, :, :, :],
                                               [2, 0, 1]), -1)
        _input_0 = tf.summary.image('conv0/input0', x[0:1])
        _conv1_0 = tf.summary.image('conv1/label0', reshape_filters(self.conv1))
        _pool1_0 = tf.summary.image('pool1/label0', reshape_filters(self.pool1))
        _pool2_0 = tf.summary.image('pool2/label0', reshape_filters(self.pool2))
        _pool3_0 = tf.summary.image('pool3/label0', reshape_filters(self.pool3))
        _pool4_0 = tf.summary.image('pool4/label0', reshape_filters(self.pool4))
        _pool5_0 = tf.summary.image('pool5/label0', reshape_filters(self.pool5))
        _input_4 = tf.summary.image('conv0/input4',
                                    [x[4 * (BATCH_SIZE / NUM_LABELS)]])
        _conv1_4 = tf.summary.image('conv1/label4', reshape_filters(self.conv1, 4))
        _pool1_4 = tf.summary.image('pool1/label4', reshape_filters(self.pool1, 4))
        _pool2_4 = tf.summary.image('pool2/label4', reshape_filters(self.pool2, 4))
        _pool3_4 = tf.summary.image('pool3/label4', reshape_filters(self.pool3, 4))
        _pool4_4 = tf.summary.image('pool4/label4', reshape_filters(self.pool4, 4))
        _pool5_4 = tf.summary.image('pool5/label4', reshape_filters(self.pool5, 4))

        self.summary_op = tf.summary.merge([_loss, _reg,
                                            _grad_norm,
                                            _var_norm,
                                            _logits, _t_acc, _input_0, _conv1_0,
                                            _pool1_0, _pool2_0, _pool3_0,
                                            _pool4_0, _pool5_0, _input_4,
                                            _conv1_4, _pool1_4, _pool2_4,
                                            _pool3_4, _pool4_4, _pool5_4])

        self.saver = tf.train.Saver()


# Init batcher, model, session.
batcher = Batcher(DATA_FP, LABELS_FP, True)
model = Model()
sess = tf.Session()

# Summary writer, acc ops.
writer = tf.summary.FileWriter(LOGDIR, sess.graph, flush_secs=30)

_acc_plhlds = [tf.placeholder(tf.float32, name='acc_' + str(i)) for i in range(NUM_LABELS + 1)]
_accs = [tf.summary.scalar('acc/' + str(i), _acc_plhlds[i]) for i in range(NUM_LABELS + 1)]
_acc_sums = tf.summary.merge(_accs)

# Training.
step = 0
sess.run(tf.global_variables_initializer())
if RESTORE:
    print 'RESTORING'
    ckpt = tf.train.get_checkpoint_state(CKPTS_RESTORE)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        step = sess.run(model.global_step)
    else:
        print 'restore failed'

IPython.embed()

try:
    for i in range(step, NUM_STEPS):
        if i > EVEN_CUTOFF:
            batcher.even_batch = False
        if step % VALIDATION_EVERY_N == 0 and step > 1:
            print 'Validation'
            model.saver.save(sess, CKPTS_SAVE, global_step=sess.run(model.global_step))
            batch_data, batch_arrays = batcher.get_validation_batch()
            inputs = {model.x: batch_arrays, model.is_training: False}
            preds = sess.run(model.preds, inputs)
            accs = get_accuracies(sess, preds, batch_data['level'].as_matrix())
            accuracy_sum = get_accuracy_summaries(sess, accs, _acc_sums, _acc_plhlds)

            writer.add_summary(tf.Summary.FromString(accuracy_sum),
                               sess.run(model.global_step))

            writer.flush()

        batch_data, batch_arrays = batcher.get_batch()
        print "Got batch"
        print batch_data['level'].values
        inputs = {
            model.x: batch_arrays,
            model.labels: np.stack(batch_data['labels'].values)
        }

        if step % SUMMARIES_EVERY_N == 0 and step > 1:
            print 'Summary'
            fetched = sess.run([model.summary_op], inputs)
            writer.add_summary(tf.Summary.FromString(fetched[0]),
                               sess.run(model.global_step))
            writer.flush()

        print 'Training'
        fetched = sess.run([model.train, model.total_loss, model.preds], inputs)
        print 'Loss: %f' % fetched[1]
        print 'Predictions:'
        print fetched[2]

        step += 1
        print step
except KeyboardInterrupt:
    pass

IPython.embed()
