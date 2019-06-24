#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 下午3:49
# @Author  : Aries
# @Site    :
# @File    : batch_normallization.py
# @Software: PyCharm
'''
在tensorflow中同样要避免过度拟合
这里我们介绍四种正则化技术:
1.提前停止
2.l1 & l2正则化
3.dropout
4.最大范数正则化
'''
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, fully_connected, l1_regularizer

from command.DataUtils import get_serialize_data
from command.SaveUtils import save

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 300
n_hidden3 = 300
n_hidden4 = 300
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
bn_params = {
    'is_training': is_training,
    'decay': 0.99,
    'updates_collections': None
}

'''
初始化时候将权重做为参数加入,就是l1范数正则化,对于多层升级网络来说实现权重的追加是比较困难的
所幸tensorflow有API的支持:
1.构建变量(eg:get_variadle()和full_connected())在构建每一个变量时候可以接受一个参数
2.你可以把权重作为参数传入并且返回相应的正则化损失函数(支持l1 & l2 & l1+l2)

这里我们在全局控制arg_scope中加入  weights_regularizer
'''
with arg_scope(
        [fully_connected],
        normalizer_fn=batch_norm,
        normalizer_params=bn_params,
        weights_regularizer=l1_regularizer(scale=0.01)):
    hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
    hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
    hidden3 = fully_connected(hidden2, n_hidden3, scope='hidden3')
    hidden4 = fully_connected(hidden3, n_hidden4, scope='hidden4')
    logits = fully_connected(hidden4, n_outputs, activation_fn=None, scope='outputs')



'''
将正则化损失函数加入到成本函数中
'''
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name='base_loss')
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_loss, name='loss')

learning_rate = 0.1

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    traing_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver

'''
execut
------------------------------------------------------------------------------------------------------------------------
'''
n_epochs = 40
batch_size = 500

mnist = get_serialize_data('mnist', 1)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(traing_op, feed_dict={is_training: True, X: X_batch, y: y_batch})
        accuracy_sorce = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y: mnist.test.labels})
        if epoch % 10 == 0:
            print(accuracy_sorce)
    save(sess, './batch_normallization/final_model')
