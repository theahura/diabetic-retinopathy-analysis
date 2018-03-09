"""
Batcher, model, training.

TODO:
    - try mse
    - add precision, recall, auc, and one off/two off accuracy
    - add test set in
    - try combining multiple eyes
    - try resnet/inception/capsule net
    - multilabel classification with new data
    - add brightness, translation, contrast variance, resolution variance
"""

import os
import random

import numpy as np
import pandas
from PIL import Image, ImageChops, ImageOps, ImageEnhance
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading

import IPython

import models

rng = np.random.RandomState(128)
random.seed(a=128)
tf.set_random_seed(128)

# Globals, hyperparameters.
NUM_STEPS = 80000
EVEN_CUTOFF = NUM_STEPS/100 * 60
SUMMARIES_EVERY_N = 100
VALIDATION_EVERY_N = 200

MSE_WEIGHT = 1.0
CX_WEIGHT = 2.0
L2_REG = 0.0002*2
LEAKY = 0.5
MOMENTUM = 0.9
LR_SCALE = 20.00
BOUNDARIES = [NUM_STEPS / 100 * i for i in [30, 50, 85, 95]]
VALUES = [i * LR_SCALE for i in [0.001, 0.0005, 0.0001, 0.00001, 0.000001]]

NUM_BATCH_THREADS = 5
LOADED_BATCH = 7
BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
TRANSLATE = False
ROTATE = True
FLIP = True

LABELS_FP = './trainLabels.csv'
DATA_FP = './train'
CKPTS_RESTORE = './ckpts'
LOGDIR = './logs/'

RESTORE = True
NUM_LABELS = 5

MODEL = 'rn'

fname = 'batchsize-%d_l2-%f_lr-%f-train-%s-%f_cutoff-%d_leaky-%f_loss-%s_weights-%f-%f_GAP_%s' % (
    BATCH_SIZE, L2_REG, LR_SCALE, 'nesterov', MOMENTUM, EVEN_CUTOFF, LEAKY,
    'Kappa+Crossent', MSE_WEIGHT, CX_WEIGHT, MODEL)

CKPTS_SAVE = CKPTS_RESTORE + '/' + fname


def get_accuracies(preds, labels):
    accs = np.zeros(NUM_LABELS + 1)
    counts = np.append(np.bincount(labels, minlength=NUM_LABELS), len(labels))
    for p, l in zip(preds, labels):
        if p == l:
            accs[l] += 1
            accs[-1] += 1
    accs /= counts

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

    def __init__(self, data_fp, label_fp, training=False):

        # Set initial vars.
        self.even_batch = True
        self.next_batch = []
        self.next_val = []
        self.next_test = []
        self.index = 0
        self.index_lock = threading.Lock()

        # Get initial data.
        files = [os.path.join(data_fp, f) for f in os.listdir(data_fp)]

        self.data = pandas.read_csv(label_fp)

        # Add filepath.
        files = sorted(files, key=lambda f: (int(f.split('/')[-1].split('_')[0]),
                                             f.split('/')[-1].split('_')[1]))
        self.data['filename'] = files

        # Randomize.
        self.data = self.data.sample(frac=1, random_state=rng)

        # Get onehots.
        onehots = np.eye(NUM_LABELS, dtype=int)

        def to_onehot(i):
            return onehots[i]

        self.data['labels'] = self.data['level'].map(to_onehot)

        if training:
            # Set the validation set to the last fifth of the training data.
            val_slice = len(self.data)/5
            self.val_data = self.data.iloc[-val_slice:]
            self.data = self.data.iloc[:-val_slice]
            print 'Getting validation arrays'
            start_daemon_thread(self._get_val_batch, (VAL_BATCH_SIZE,))

            for _ in range(NUM_BATCH_THREADS):
                start_daemon_thread(self._get_batch, args=(BATCH_SIZE,))
        else:
            for _ in range(NUM_BATCH_THREADS):
                start_daemon_thread(self._get_ordered_batch, args=(BATCH_SIZE,))

    def _get_distribution_batch(self, size):
        """Samples a minibatch according to the true distribution."""
        batch_data = self.data.sample(size, random_state=rng)
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

    def _get_ordered_batch(self, size):
        """Samples a minibatch evenly across classes, in order."""
        while True:
            self.index_lock.acquire()
            if len(self.next_test) >= LOADED_BATCH:
                self.index_lock.release()
                continue
            if self.index >= len(self.data):
                return
            start = self.index
            end = min(self.index + size, len(self.data))
            self.index += size
            self.index_lock.release()
            batch_data = self.data.iloc[start:end]
            batch_arrays = np.stack(batch_data['filename'].map(open_files).values)
            self.next_test.append((batch_data, batch_arrays/255.0))

    def _get_batch(self, size):
        """Preloads LOADED_BATCH train minibatches."""
        while True:
            if len(self.next_batch) >= LOADED_BATCH:
                continue
            if self.even_batch:
                data, arrays = self._get_even_batch(size)
            else:
                data, arrays = self._get_distribution_batch(size)
            self.next_batch.append((data, arrays))

    def _get_val_batch(self, size):
        """Preloads two validation minibatches."""
        while True:
            if len(self.next_val) >= 2:
                continue
            indices = np.random.choice(len(self.val_data), size, replace=False)
            batch_data = self.val_data.iloc[indices]
            batch_arrays = np.stack(batch_data['filename'].map(open_files).values)

            self.next_val.append((batch_data, batch_arrays/255.0))

    def get_test_batch(self):
        while not len(self.next_test):
            if self.index >= len(self.data):
                return None, None
        return self.next_test.pop(0)

    def get_batch(self):
        while not len(self.next_batch):
            pass
        return self.next_batch.pop(0)

    def get_validation_batch(self):
        while not len(self.next_val):
            pass
        return self.next_val.pop(0)


def quad_kappa_loss(y, t, y_pow=2, eps=1e-15):
    t = tf.cast(t, tf.float32)
    ratings = np.tile(np.arange(0, NUM_LABELS)[:, None], reps=(1, NUM_LABELS))
    ratings_sq = (ratings - ratings.T)**2
    weights = ratings_sq / (float(NUM_LABELS) - 1)**2

    y_ = y ** y_pow
    y_norm = y_ / (eps + tf.reduce_sum(y_, axis=1)[:, None])

    hist_rater_a = tf.reduce_sum(y_norm, axis=0)
    hist_rater_b = tf.reduce_sum(t, axis=0)

    conf_mat = tf.matmul(tf.transpose(y_norm), t)

    nom = tf.reduce_sum(weights * conf_mat)
    expected_probs = tf.matmul(hist_rater_a[:, None], hist_rater_b[None, :])
    denom = tf.reduce_sum(weights * expected_probs) / BATCH_SIZE
    return -(1 - nom / denom)


# Create network.
class Model(object):

    def __init__(self):
        self.x = x = tf.placeholder(tf.float32, shape=(None, 512, 512, 3), name='x')
        self.is_training = is_training = tf.placeholder_with_default(True, shape=())

        def leaky(x):
            return tf.nn.leaky_relu(x, LEAKY)

        if MODEL == 'basic':
            net, endpoints = models.basic_model(x, leaky, L2_REG, is_training)
        elif MODEL == 'ir':
            net, endpoints = models.inception_resnet(x, is_training, leaky)
        elif MODEL == 'rn':
            net, endpoints = models.resnet(x, is_training, leaky)

        self.endpoints = endpoints

        self.feats = slim.conv2d(net, num_outputs=1024, kernel_size=3,
                                 activation_fn=leaky,
                                 weights_regularizer=slim.l2_regularizer(L2_REG))
        self.gap = tf.reduce_mean(self.feats, [1, 2])

        net = slim.dropout(self.gap, 0.5, is_training=is_training)
        self.logits = slim.fully_connected(net, NUM_LABELS,
                                           activation_fn=None,
                                           weights_regularizer=slim.l2_regularizer(L2_REG),
                                           scope='fc')
        self.sftmx = tf.nn.softmax(self.logits)
        self.preds = tf.cast(tf.argmax(self.sftmx, 1), tf.int64)

        # Losses.
        self.labels = tf.placeholder(tf.int64, name='labels')
        self.levels = tf.placeholder(tf.int64, name='levels')
        self.mse_loss = tf.losses.mean_squared_error(self.levels, self.preds)

        self.kappa = quad_kappa_loss(self.sftmx, self.labels)
        self.crossent_loss = tf.losses.softmax_cross_entropy(self.labels,
                                                             self.logits)
        clipped_cx = tf.clip_by_value(self.crossent_loss, 0.8, 10**3)
        self.norm_loss = tf.losses.get_regularization_losses()
        self.total_loss = tf.reduce_mean(CX_WEIGHT*clipped_cx +
                                         MSE_WEIGHT*self.kappa +
                                         self.norm_loss)

        # Optimizer.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.piecewise_constant(self.global_step, BOUNDARIES,
                                              VALUES)
        self.op = tf.train.MomentumOptimizer(self.lr, MOMENTUM,
                                             use_nesterov=True)

        tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(self.total_loss, tvs)
        gvs = list(zip(grads, tvs))
        #self.train = self.op.apply_gradients(gvs, global_step=self.global_step)
        self.train = slim.learning.create_train_op(self.total_loss, self.op,
                                                   self.global_step)

        # Model Summaries.
        _loss = tf.summary.scalar('model/tot_loss', tf.reduce_mean(self.total_loss))
        _reg = tf.summary.scalar('model/reg_losses', tf.reduce_mean(self.norm_loss))
        _cx = tf.summary.scalar('model/cx_losses', self.crossent_loss)
        _mse = tf.summary.scalar('model/mse_losses', self.mse_loss)
        _kappa = tf.summary.scalar('model/kappa_losses', self.kappa)
        _grad_norm = tf.summary.scalar('model/grad_norm', tf.global_norm(grads))
        _var_norm = tf.summary.scalar('model/var_norm', tf.global_norm(tvs))
        _logits = tf.summary.histogram('model/preds', self.preds)
        _t_acc = tf.summary.scalar(
            'acc/train_acc', tf.contrib.metrics.accuracy(self.preds, self.levels))

        _grads = [tf.summary.histogram('model/' + v.name, g) for g, v in gvs
                  if g is not None]

        def reshape(layer, label_index):
            return tf.expand_dims(tf.transpose(layer[label_index, :, :, :],
                                               [2, 0, 1]), -1)

        def heatmap(index, x):
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
            return tf.image.resize_bilinear(cam, [512, 512])

        min_i = tf.argmin(self.levels)
        max_i = tf.argmax(self.levels)
        _input_0 = tf.summary.image('conv0/input0', [x[min_i]])
        _hm0 = tf.summary.image('conv0/hm0', heatmap(min_i, x))
        _input_4 = tf.summary.image('conv0/input4', [x[max_i]])
        _hm4 = tf.summary.image('conv0/hm4', heatmap(max_i, x))

        self.summary_op = tf.summary.merge([_loss, _reg, _cx, _mse, _kappa,
                                            _grad_norm, _var_norm,
                                            _logits, _t_acc, _input_0,
                                            _input_4, _hm0, _hm4] +
                                           _grads)

        self.saver = tf.train.Saver()


def main():
    # Init batcher, model, session.
    batcher = Batcher(DATA_FP, LABELS_FP, True)

    model = Model()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Summary writer, acc ops.
    writer = tf.summary.FileWriter(LOGDIR, sess.graph, flush_secs=30)

    _acc_plhlds = [tf.placeholder(tf.float32, name='acc_' + str(i)) for
                   i in range(NUM_LABELS + 1)]
    _accs = [tf.summary.scalar('acc/' + str(i), _acc_plhlds[i]) for
             i in range(NUM_LABELS + 1)]
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
                preds = []
                labels = []
                for j in range(10):
                    batch_data, batch_arrays = batcher.get_validation_batch()
                    inputs = {model.x: batch_arrays, model.is_training: False}
                    preds += sess.run(model.preds, inputs).tolist()
                    labels += batch_data['level'].tolist()

                accs = get_accuracies(preds, labels)
                print 'Batch Data'
                print batch_data['level'].values
                print 'Predictions'
                print preds
                print 'Accuracies'
                print accs
                accuracy_sum = get_accuracy_summaries(sess, accs, _acc_sums, _acc_plhlds)

                writer.add_summary(tf.Summary.FromString(accuracy_sum),
                                   sess.run(model.global_step))

                writer.flush()

            batch_data, batch_arrays = batcher.get_batch()
            print "Got batch"
            print batch_data['level'].values
            inputs = {
                model.x: batch_arrays,
                model.labels: np.stack(batch_data['labels'].values),
                model.levels: np.stack(batch_data['level'].values)
            }

            if step % SUMMARIES_EVERY_N == 0 and step > 1:
                print 'Summary'
                fetched = sess.run([model.summary_op], inputs)
                writer.add_summary(tf.Summary.FromString(fetched[0]),
                                   sess.run(model.global_step))
                writer.flush()

            print 'Training'
            fetched = sess.run([model.train, model.total_loss, model.preds],
                               inputs)
            print 'Loss: %f' % fetched[1]
            print 'Predictions:'
            print fetched[2]

            step += 1
            print step
    except KeyboardInterrupt:
        pass

    IPython.embed()


if __name__ == '__main__':
    main()
