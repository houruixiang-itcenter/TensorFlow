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
from command.SaveUtils import save

'''
在神经网络中,为了缓和梯度消失/爆炸,提高算法的收敛效率,要对数据进行零中心化和归一化操作

(X-均值)/标准差 进行零中心化和归一化的操作
'''
'''
方式一:
-------------
tensorflow中有提供一个方法
批量归一化的操作需要你自己计算均值和方差,然后做为参数传给这个方法,
而且你自己必须确定缩放和偏移

方式二:
-------------
tensorflow还有一个比较简单的方法,batch_normal函数,他提供了所有的参数,可以直接调用,
或者告诉fully_connected()函数去调用他
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
bn_params = {
	'is_training': is_training,
	'decay': 0.99,
	'updates_collections': None
}

'''
1.fully_connected()来创建层
2.调用激活函数之前通过调用batch_norm()函数(通过参数bn_params)来进行输入归一化


注意:
默认情况下-----------------
batch_norm()只中心化,归一化和对输入进行偏移操作,但是并不缩放(缩放值恒为1);这样对于没有激活函数或者用ReLU激活函数的层
是有效果的,但是对于其他的激活函数,你需要设置"scale",即将bn_params设置为True
'''
# hidden1 = fully_connected(X, n_hidden1, scope='hidden1',
#                           normalizer_fn=batch_norm, normalizer_params=bn_params)
# hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2',
#                           normalizer_fn=batch_norm, normalizer_params=bn_params)
# logits = fully_connected(hidden2, n_outputs, activation_fn=None, scope='outputs',
#                          normalizer_fn=batch_norm, normalizer_params=bn_params)

'''
你可能已经注意到,定义前三层是重复的,因为有几个参数是相同的
为了避免一直重复参数,你可以用arg_scope()方法构建一个参数范围:
第一个参数是一个函数列表,其他参数会自动传给这些函数;

就是说全局设定完后 不需要重复一条一条的设定

下面看code
'''
with arg_scope(
		[fully_connected],
		normalizer_fn=batch_norm,
		normalizer_params=bn_params):
	hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
	hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
	hidden3 = fully_connected(hidden2, n_hidden3, scope='hidden3')
	hidden4 = fully_connected(hidden3, n_hidden4, scope='hidden4')
	logits = fully_connected(hidden4, n_outputs, activation_fn=None, scope='outputs')
'''
这种写法在10层以上的神经网络中,可读性会大大提高
'''

'''
接下来和之前ANN一样:
1.定义成本函数
2.构建优化器,让他最小化成本函数
3.定义评估操作
4.创建Saver
'''
with tf.name_scope('loss'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01
with tf.name_scope('train'):
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	'''
	minimize:负责计算和应用梯度
	'''
	traing_op = optimizer.minimize(loss)

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
	save(sess,'./batch_normallization/final_model')


