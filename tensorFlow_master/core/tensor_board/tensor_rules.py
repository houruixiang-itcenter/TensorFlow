#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/11 下午9:00
# @Author  : Aries
# @Site    : 
# @File    : tensor_rules.py
# @Software: PyCharm
'''
模块化
'''
import tensorflow as tf

'''
在我看来 这个模块化和java中的递归差不多

假设你想要计算亮哥修正单元(ReLU)之后的图
如果值是正值则输出其值,如果是负数则返回0

下面是冗余版的code
'''
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')

w1 = tf.Variable(tf.random_normal((n_features, 1)), name='weights1')
w2 = tf.Variable(tf.random_normal((n_features, 1)), name='weights2')
b1 = tf.Variable(0.0, name='bias1')
b2 = tf.Variable(0.0, name='bias2')

z1 = tf.add(tf.matmul(X, w1), b1, name='z1')
z2 = tf.add(tf.matmul(X, w2), b2, name='z2')

relu1 = tf.maximum(z1, 0.0, name='relu1')
relu2 = tf.maximum(z2, 0.0, name='relu2')

output = tf.add(relu1, relu2, name='output')

'''
优化之后
'''


def relu(X):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal((w_shape)), name='weights1')
    b = tf.Variable(0.0, name='bias')
    z = tf.add(tf.matmul(X, w), b, name='z')
    return tf.add(z, 0.0, name='relu')


n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name='output')
