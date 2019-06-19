#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/19 下午1:57
# @Author  : Aries
# @Site    : 
# @File    : dnn_gradient.py
# @Software: PyCharm
'''
在深度网络--梯度消失 and 梯度爆炸
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected

'''
深度学习,由于激活函数选择不恰当或者随着神经网络层的叠加,当反向更新权重的过程中会出现梯度消失或者梯度爆炸
原因:链式相乘导致
具体会之后同步博客,这里不做过多的赘述

解决梯度消失和爆炸问题可以从两方面解决:
1.权重初始化使用Xavier和He初始化,使得每一层的权重初始化方差一致
2.从激活函数入手:
- softmax的偏导数最大值为1/4,链式相乘导致反向更新权重的梯度越来越小
- ReLu:当z大于0时,偏导为1,这样不会对底层的神经元造成梯度稀释或者梯度爆炸,但是当输出小于0的时候,部分神经元不会被激活
- 基于ReLu的弊端,输入小于0时候梯度为0,推出ReLU变种:leaky ReLU,RReLU;和ELU
'''

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')


def neuron_layer(X, n_neurons, name, activation=None):
    '''
    创建一个作用域
    '''
    with tf.name_scope(name):

        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(X, W) + b
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z


'''
tensorflow给你提供了一个elu()函数来构建神经网络,当调用fully_connected()函数时候,可以很简单的设置activation_fn参数
'''
with tf.name_scope('dnn'):
    # 300
    hidden1 = fully_connected(X, n_hidden1, activation_fn=tf.nn.elu)
    # 100
    hidden2 = fully_connected(hidden1, n_hidden2, activation_fn=tf.nn.elu)
    # 10
    logits = fully_connected(hidden2, n_outputs, 'outputs')

'''
tensorflow中没有leaky ReLU函数的预定义函数,但是也可以简单的定义为:
'''


def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)


with tf.name_scope('dnn'):
    # 300
    hidden1 = fully_connected(X, n_hidden1, activation_fn=leaky_relu)
    # 100
    hidden2 = fully_connected(hidden1, n_hidden2, activation_fn=leaky_relu)
    # 10
    logits = fully_connected(hidden2, n_outputs, 'outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    traing_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver
