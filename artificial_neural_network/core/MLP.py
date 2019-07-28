#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/14 下午12:36
# @Author  : Aries
# @Site    : 
# @File    : MLP.py
# @Software: PyCharm
'''
多层感知器:MLP
'''
import os
from sklearn.metrics import accuracy_score

import numpy as np
from tensorflow.contrib.learn.python import learn
from command.DataUtils import get_mnist_train, get_mnist_test, serialize_data, get_serialize_data
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
使用tensorflow的高级API来训练MLP

用tensorflow训练MLP的最简单方式是使用他的高级API TF.Learn,这和Scikit的API非常类似

用DNNClassifier训练一个有这任意隐藏层,并包含一个用来计算类别概率的softmax输出层的深度神经网络

eg:下面训练一个用于分类的两个隐藏层(一个300个神经元,另一个100个),以及一个softmax输出层的具有10个神经元的DNN
'''
X_train, Y_train = get_mnist_train()
feature_cloumns = learn.infer_real_valued_columns_from_input(X_train)

dnn_clf = learn.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_cloumns)
try:
    if get_serialize_data('dnn_clf', mode=1) is None:
        serialize_data(dnn_clf, 'dnn_clf', mode=1)
except FileNotFoundError as e:
    serialize_data(dnn_clf, 'dnn_clf', mode=1)
'''
batch_size:小批量X的size
steps:循环次数
'''
dnn_clf.fit(X_train, Y_train.astype(np.int), batch_size=50, steps=40000)

'''
评估准确率
'''
X_test, Y_test = get_mnist_test()
y_pred = list(dnn_clf.predict(X_test))
score = accuracy_score(Y_test, y_pred)

print(score)
'''
库中含有评估模型的函数,不需要向上述那样使用Scikit模型来进行评估
'''
score = dnn_clf.evaluate(X_test, Y_test.astype(np.int))
print(score)


'''
在幕后,DNNClassifier类基于ReLU激活函数(我们可以通过设置activation_fn超参数来调整),创建所有的神经元层次;输出层次依赖于softmax函数,
成本函数就是交叉熵
'''
'''
TF.Learn API还是比较新的
'''

