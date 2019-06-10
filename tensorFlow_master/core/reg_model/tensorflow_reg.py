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

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from command.DataUtils import serialize_data

'''
需要加入这一行  这样http证书就不会错误了
'''
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

'''
回顾之前的scikit可知 标准方程是成本函数对权重求偏导  得到最合适的权重(即此时成本函数最小)
'''
housing = fetch_california_housing()
# housing = load_housing_data()
print(type(housing))
m, n = housing.data.shape
# 增加偏置项
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
serialize_data(housing, 'housing')
serialize_data(housing_data_plus_bias, 'housing_data_plus_bias')
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
