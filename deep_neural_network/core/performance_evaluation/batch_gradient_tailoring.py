#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 下午3:49
# @Author  : Aries
# @Site    :
# @File    : batch_normallization.py
# @Software: PyCharm
'''
用tensorflow实现批量归一化
'''
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, fully_connected

from command.DataUtils import get_serialize_data

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
bn_params = {
	'is_training': is_training,
	'decay': 0.99,
	'updates_collections': None
}

with arg_scope(
		[fully_connected],
		normalizer_fn=batch_norm,
		normalizer_params=bn_params):
	hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
	hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
	logits = fully_connected(hidden2, n_outputs, activation_fn=None, scope='outputs')

with tf.name_scope('loss'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	loss = tf.reduce_mean(xentropy, name='loss')

'''
还有一种减轻梯度爆炸的方式叫做梯度裁剪,从而保证在梯度反向传递过程中梯度不超过阈值
虽然大家多倾向于梯度的归一化,但是这个方式也有必要做一下了解


minimize:负责计算和应用梯度
现在改为:
1.调用优化器的compute_gradients()
2.然后调用clip_by_value()方法创建一个剪裁梯度的操作
3.最后调用apply_gradients()方法应用剪裁后的梯度

'''
learning_rate = 0.01
with tf.name_scope('train'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	grads_and_vars = optimizer.compute_gradients(loss)
	'''
	grad:梯度
	-1.0 & 1.0:阈值 可以作为超参数进行网格搜索
	'''
	capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars]
	traing_op = optimizer.apply_gradients(capped_gvs)

with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver

'''
接下来进行执行阶段

但是有一点不同,无论何时你运行一个依赖于batch_norm层的操作
你都需要设置is_training占位符为True或者False
'''

'''
小批次梯度下降
定义epoch数量,以及小批次的大小
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
		print(accuracy_sorce)
