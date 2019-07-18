#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 下午10:38
# @Author  : Aries
# @Site    :
# @File    : Dnn-master.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

'''
定义训练轮数
'''
traing_steps = 30000

'''
定义输入的数据和对应的标签并在for循环内进行填充
batch数据输入
'''
data = []
label = []
for i in range(200):
	x1 = np.random.uniform(-1, 1)
	x2 = np.random.uniform(0, 2)
	'''
	策略:对x1 and x2 进行判断,如果落在原点为中心1为半径的圆内,label = 0
		反之为1
	'''
	if x1**2 + x2**2 <= 1:
		data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
		label.append(0)
	else:
		data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
		label.append(1)

'''
numpy的hstack()函数用于再水平方向将元素堆起来
函数圆形 numpy.hstack(tup) tup 可以是元组,列表或者numpy数组
reshape用于反转
'''
data = np.hstack(data).reshape(-1, 2)
label = np.hstack(label).reshape(-1, 1)

'''
定义完成前馈传递的隐藏层
'''


def hidden_layer(input, w1, b1, w2, b2, w3, b3):
	layer1 = tf.nn.relu(tf.matmul(input, w1) + b1)
	layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
	return tf.matmul(layer2, w3) + b3


x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-output')

'''
定义权重参数和偏置参数
'''
w1 = tf.Variable(tf.truncated_normal([2, 10], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[10]))
w2 = tf.Variable(tf.truncated_normal([10, 10], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))
w3 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[1]))

'''
用len记录data的长度
'''
sample_size = len(data)

'''
得到隐藏层前向传播结果
'''
y = hidden_layer(x, w1, b1, w2, b2, w3, b3)

'''
自定义损失函数
'''
error_loss = tf.reduce_sum(tf.pow(y_ - y, 2)) / sample_size
tf.add_to_collection('loss', error_loss)

regularizer = layers.l2_regularizer(0.01)
regularization = regularizer(w1) + regularizer(w2) + regularizer(w3)
tf.add_to_collection('loss', regularization)

'''
get_collection()根据name,获取所有的损失值进行加运算
'''
loss = tf.add_n(tf.get_collection('loss'))

'''
定义一个优化器进行梯度更新
学习率固定为0.01
'''
traing_op = tf.train.AdamOptimizer(0.01).minimize(loss)

'''
exect()
'''
with tf.Session() as sess:
	'''
	初始化tf变量
	'''
	tf.global_variables_initializer().run()
	
	'''
	进行30000次循环
	'''
	for i in range(traing_steps):
		sess.run(traing_op, feed_dict={x: data, y_: label})
		
		'''
		每隔2000次输出一次loss值
		'''
		if i % 2000 == 0:
			loss_value = sess.run(loss, feed_dict={x: data, y_: label})
			print('afer %d step ,loss value is %f' % (i, loss_value))
