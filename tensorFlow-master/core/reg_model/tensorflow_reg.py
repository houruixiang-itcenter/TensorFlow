#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/6 下午4:40
# @Author  : Aries
# @Site    :
# @File    : tensorflow_reg.py
# @Software: PyCharm
'''
TensorFlow中线性回归
'''
import os

import numpy
import numpy as np
import tensorflow as tf
from pandas import DataFrame
from sklearn.datasets import fetch_california_housing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
回顾之前的scikit可知 标准方程是成本函数对权重求偏导  得到最合适的权重(即此时成本函数最小)
'''
housing = fetch_california_housing()
print(type(housing))
m, n = housing.data.shape
# 增加偏置项
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
'''
构建计算机图
'''
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='x')
Y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
'''
tf.transpose: 矩阵的转置
tf.matmul:  乘法
tf.matrix_inverse: -1次方
'''
thera = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), Y)

with tf.Session() as session:
    thera_value = thera.eval()
print(thera_value)

