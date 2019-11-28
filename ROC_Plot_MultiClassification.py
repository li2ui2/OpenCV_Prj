# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import train
from sklearn.preprocessing import label_binarize
import numpy as np
from scipy import interp
from itertools import cycle
# 值计算
X_test = train.X_test
y_test = train.y_test
y_test= label_binarize(y_test, classes=[i for i in range(42)]) # 样本标签二值化矩阵
model = joblib.load('./models/svm_classifier42.model')
y_score = model.decision_function(X_test) # 样本置信值矩阵
fpr = dict()
tpr = dict()
roc_auc = dict()

'''
#方法一
#计算每一类的ROC
n_classes = y_score.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
'''

# 方法二
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"], 'b--',
         label='ROC (area = %0.6f)' % roc_auc["micro"],
         linestyle=':',lw=2)
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='cornflowerblue', linestyle=':', linewidth=4)

# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC to multi-class')
plt.legend(loc="lower right")
plt.show()