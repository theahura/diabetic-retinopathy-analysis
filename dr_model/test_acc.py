"""Gets test accuracies.
TODO:
    add spec, sens

"""

import json
import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf

import IPython

from dr import Batcher, Model


LABELS_FP = './testLabels.csv'
DATA_FP = './test'
CKPTS_RESTORE = './ckpts'
NUM_LABELS = 5

TEST_PRED_FP = './tests/test_preds'
TEST_LABEL_FP = './tests/test_labels'
TEST_LOGIT_FP = './tests/test_logits'

def get_accuracies(preds, labels):
    accs = np.zeros(NUM_LABELS + 1)
    for p, l in zip(preds, labels):
        if p == l:
            accs[l] += 1
            accs[-1] += 1

    return accs


def get_actual_accuracies(preds, labels):
    accs = 0
    for p, l in zip(preds, labels):
        if (p == 0 and l == 0) or (p > 0 and l > 0):
            accs += 1

    return accs


batcher = Batcher(DATA_FP, LABELS_FP)
model = Model()
sess = tf.Session()

print 'RESTORING'
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(CKPTS_RESTORE)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    model.saver.restore(sess, ckpt.model_checkpoint_path)
    step = sess.run(model.global_step)
else:
    raise ValueError

total_correct = 0.
total_actual_correct = 0
total_seen = 0

lo = []
p = []
l = []

while True:
    batch_data, batch_arrays = batcher.get_test_batch()
    if batch_data is None:
        break

    inputs = {model.x: batch_arrays, model.is_training: False}
    preds, logits = sess.run([model.preds, model.sftmx], inputs)
    preds = preds.tolist()
    logits = logits.tolist()
    labels = batch_data['level'].tolist()
    p += preds
    l += labels
    lo += logits
    total_correct += get_accuracies(preds, labels)
    total_actual_correct += get_actual_accuracies(preds, labels)
    total_seen += len(batch_data)

    print 'Accuracies'
    print total_correct/(total_seen * 1.0)
    print 'Actual accuracy'
    print total_actual_correct/(total_seen * 1.0)

with open(TEST_PRED_FP, 'w') as f:
    json.dump(p, f)

with open(TEST_LABEL_FP, 'w') as f:
    json.dump(l, f)

with open(TEST_LOGIT_FP, 'w') as f:
    json.dump(lo, f)
