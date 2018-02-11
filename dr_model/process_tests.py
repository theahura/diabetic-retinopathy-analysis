"""
Take test outputs and do some stats.
"""

import json
import sklearn.metrics
import IPython

with open('test_preds') as f:
    p = json.load(f)

with open('test_labels') as f:
    la = json.load(f)

with open('test_logits') as f:
    lo = json.load(f)

# Get precision, recall, f1, confusion
confusion = sklearn.metrics.confusion_matrix(la, p)
report = sklearn.metrics.classification_report(la, p)

print confusion
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

print sklearn.metrics.roc_auc_score(roc_la, roc_logit_prob, average='micro')
print sklearn.metrics.roc_auc_score(roc_la, roc_logit_prob, average='macro')

IPython.embed()
