#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 下午3:49
# @Author  : Aries
# @Site    :
# @File    : batch_normallization.py
# @Software: PyCharm
'''
最大范数正则化
'''
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, fully_connected, l1_regularizer

from command.DataUtils import get_serialize_data
from command.SaveUtils import save

'''
最大范数正则化:
对于每一个神经元,包含一个传入连接权重w满足其l2范数<=r,r是最大范数的超参数;

降低r会增加正则化的数目,同时帮助减少过度拟合;最大范数正则化可以同时帮助缓解梯度消失/爆炸问题(如果不使用批量归一化)
'''

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 300
n_hidden3 = 300
n_hidden4 = 300
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
'''
tensorflow没有提供现成的最大范数正则化器,但是实现起来并不难
下面代码构建一个节点clip_weights,该节点会沿着第二个轴削减weights变量,从而使每一个行向量的最大范数为1
'''
# threshold = 1.0
# clipped_weights = tf.clip_by_norm(weights, clip_norm=threshold, axes=1)
# clip_weights = tf.assign(weights, clipped_weights)


with arg_scope(
        [fully_connected],
        normalizer_fn=batch_norm):
    hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
    hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
    hidden3 = fully_connected(hidden2, n_hidden3, scope='hidden3')
    hidden4 = fully_connected(hidden3, n_hidden4, scope='hidden4')
    logits = fully_connected(hidden4, n_outputs, activation_fn=None, scope='outputs')

# with tf

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
            # clip_weights.eval()
        accuracy_sorce = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y: mnist.test.labels})
        if epoch % 10 == 0:
            print(accuracy_sorce)
    save(sess, './batch_normallization/final_model')
