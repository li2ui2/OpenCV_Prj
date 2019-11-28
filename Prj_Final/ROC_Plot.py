# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import train
# 值计算
X_test = train.X_test
y_test = train.y_test
model = joblib.load('./models/svm_classifier42.model')
y_score = model.decision_function(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
print(roc_auc)
# 绘图
plt.plot(fpr, tpr, 'k--', label='ROC (area = %0.2f)' % roc_auc, lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()