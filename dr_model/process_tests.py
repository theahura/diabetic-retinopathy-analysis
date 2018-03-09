"""
Take test outputs and do some stats.
"""

import json
import sklearn.metrics
import IPython


def get_actual_preds(preds, labels):
    act_p = []
    act_l = []
    for p, l in zip(preds, labels):
        p = p if p == 0 else 1
        l = l if l == 0 else 1
        act_p.append(p)
        act_l.append(l)

    return act_p, act_l


with open('tests/test_preds') as f:
    p = json.load(f)

with open('tests/test_labels') as f:
    la = json.load(f)

with open('tests/test_logits') as f:
    lo = json.load(f)

act_p, act_la = get_actual_preds(p, la)

# Get precision, recall, f1, confusion
confusion = sklearn.metrics.confusion_matrix(la, p)

act_confusion = sklearn.metrics.confusion_matrix(act_p, act_la)
TP = act_confusion[0][0]
FP = act_confusion[0][1]
FN = act_confusion[1][0]
TN = act_confusion[1][1]

print "Sensitivity: %f" % (TP*1.0/(TP + FN))
print "Specificity: %f" % (TN*1.0/(FP + TN))

report = sklearn.metrics.classification_report(la, p)

print confusion
print act_confusion
print report

# Get ROC
roc_logit_prob = []
roc_la = []
for label, logit in zip(la, lo):
    if label == 0:
        roc_logit_prob.append(logit[0])
        roc_la.append(0)
    else:
        roc_logit_prob.append(sum(logit[1:]))
        roc_la.append(1)

print "ROC"
print sklearn.metrics.roc_auc_score(roc_la, roc_logit_prob, average='micro')
print sklearn.metrics.roc_auc_score(roc_la, roc_logit_prob, average='macro')

IPython.embed()
