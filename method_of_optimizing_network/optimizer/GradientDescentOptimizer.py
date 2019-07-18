#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 下午5:38
# @Author  : Aries
# @Site    : 
# @File    : GradientDescentOptimizer.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.contrib import layers

'''
这里设置  training_rate 代表目前正在进行的训练轮数
一版这个值会随着训练的进行而同步增大
'''
training_step = tf.Variable(0)

'''
使用exponential_decay()函数设置学习率,global_step值为training_step
'''
decayed_learning_rate = tf.train.exponential_decay(0.8, training_step, 100, 0.9, staircase=True)

'''
使用一个梯度优化器,其中损失函数loss式目标函数
# '''
# learning_step = tf.train.GradientDescentOptimizer(decayed_learning_rate)\
#     .minimize(loss,globals()training_step);


# weights = tf.constant([[1.0, 2.0], [-3.0, -4.0]])
# weights = tf.constant([3.0, 4.0])
weights = tf.constant([3.0])
'''
l1 & l2 范数
'''
regularizer_l1 = layers.l1_regularizer(.5)
regularizer_l2 = layers.l2_regularizer(.5)

with tf.Session() as sess:
    print(sess.run(regularizer_l1(weights)))
    print(sess.run(regularizer_l2(weights)))
