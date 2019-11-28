# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import train
import numpy as np
from sklearn.preprocessing import label_binarize
# CMC曲线
# 需要提供confidence_values和y_test这两个变量
X_test = train.X_test
y_test_ = train.y_test
model = joblib.load('./models/svm_classifier42.model')
# 得到每个样本在每个标签下的置信值，返回的confidence_values为模型预测得到的匹配置信分数矩阵
confidence_values = model.decision_function(X_test)
# 将标签二值化,返回一个one-hot标签矩阵
y_test= label_binarize(y_test_, classes=[i for i in range(42)])
# 保存accuracy，记录rank1到rank42的准确率
test_cmc = []

# y_test为测试样本的真实标签矩阵；返回每行真实标签相对应的最大值的索引值
actual_index = np.argmax(y_test,1)
#返回每行预测标签相对应的最大值的索引值
predict_index = np.argmax(confidence_values,1)
# 若为1代表相同，一次命中；0代表不同，第一次猜测并未命中
temp = np.cast['float32'](np.equal(actual_index,predict_index))
# rank1
test_cmc.append(np.mean(temp))

# 按行降序排序，返回匹配分数值从大到小的索引值
sort_index = np.argsort(-confidence_values,axis=1)
# rank2到rank42
for i in range(sort_index.shape[1]-1):
    for j in range(len(temp)):
        if temp[j]==0:
            predict_index[j] = sort_index[j][i+1]
    temp = np.cast['float32'](np.equal(actual_index,predict_index))
    test_cmc.append(np.mean(temp))

#创建绘图对象
plt.figure()
x = np.arange(0,sort_index.shape[1])
plt.plot(x,test_cmc,color="red",linewidth=2)
plt.xlabel("Rank")
plt.ylabel("Matching Rate")
plt.legend()
plt.title("CMC Curve")
plt.show()