#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/14 下午4:28
# @Author  : Aries
# @Site    : 
# @File    : dnn_model.py
# @Software: PyCharm
'''
使用纯tensorflow训练DNN
如果想对网络的架构有更多的控制,可以使用tenwsorflow低级的API;
下面我们使用低级的API构建一个和上一节DNNClassifier相同的模型
实现一个小批次的梯度下降来训练MNIST数据集


1.建立tensorflow的计算图
2.执行阶段,具体运行这个图来训练模型
'''

'''
构建阶段
首先需要引入tensorflow库,然后是指定输入和输出的个数,并设置每层隐藏神经元的个数
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

'''
紧接着定义X,Y的占位符
X: 用作输入层
'''
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')


'''
接下来我们来创建神经网络
1.占位符节点X将用作输入层;在执行期,它每次都会被训练批次替换(注意训练批次中的所有实例将由神经网络同时处理)
2.创建两个隐藏层;两个隐藏层基本上是一样的:唯一的区别是他们和谁链接,以及每层中包含的神经元数量
3.创建一个输出层:和隐藏层不同,它会用softmax而不是ReLU做为激活函数
4.我们创建一个neuron_layer()函数来每次创建一个层;
----- 它需要的参数包括:输入,神经元数量,激活函数,层次的名字:
'''


def neuron_layer(X, n_neurons, name, activation=None):
	'''
	创建一个作用域
	'''
	with tf.name_scope(name):
		# 输出数据量的number
		'''
		获取矩阵的第二个纬度
		'''
		n_inputs = int(X.get_shape()[1])
		'''
		指定标准偏差为stddev的截断正态(高斯)分布,进行随机初始化,使用一个指定的标准偏差会让算法收敛的更快
		'''
		stddev = 2 / np.sqrt(n_inputs)
		'''
		接下来两行用来构建每个输入和每个神经元之间连接的权重
		'''
		init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
		W = tf.Variable(init, name='weights')
		b = tf.Variable(tf.zeros([n_neurons]), name='biases')
		z = tf.matmul(X, W) + b
		'''
		输出层的激活函数,可以通过传参进行选择和指定
		'''
		if activation == 'relu':
			return tf.nn.relu(z)
		else:
			return z


# todo 这里是调用构建的过程
'''
好了,我们现在有了一个创建神经元的函数了,我们可以用它来构建一个深度神经网络
第一个隐藏层需要X作为其输入
第二层则以第一层的输出作为输入
最后输出层以第二层的输出作为输入
'''

with tf.name_scope('dnn'):
	'''
	使用命名空间来保持名字的清晰
	注意: logits是经过softmax激活函数之前的神经网络的输出:基于优化的考虑,我们将在稍候处理softmax的计算
	'''
	# 300
	hidden1 = neuron_layer(X, n_hidden1, 'hidden1', activation='relu')
	# 100
	hidden2 = neuron_layer(hidden1, n_hidden2, 'hidden2', activation='relu')
	# 10
	logits = neuron_layer(hidden2, n_outputs, 'outputs')

'''
tensorflow提供了很多便利的函数创建标准神经网络层,所以通常无需定义自己的neuron_layer函数
eg:
TensorFlow的fully_connected()函数会创建全连接层,其中左右输入,都连接到该层中的所有神经元
---这个函数会创建权重和标准偏差,使用合适的初始化策略,使用ReLU激活函数(可以通过activation_fn参数来修改)


代码如下:
'''
# with tf.name_scope('dnn'):
# 	# 300
# 	hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
# 	# 100
# 	hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
# 	# 10
# 	logits = fully_connected(hidden2, n_outputs, scope='outputs', activation_fn=None)

'''
		我们已经有了神经网络模型,现在需要定义成本函数用以训练它
		这里我们回使用softmax中的交叉熵,之前我们讨论过交叉熵回处罚那些估计目标类的概率较低的模型
		tensorflow提供了很多函数来计算交叉熵,我们这里会用spare_soft_max_cross_entropy_with_logits():
		它会根据'logits'来计算交叉熵(比如通过softmax激活函数之前网络的输出)
		这会计算出一个包含每个实例的交叉熵的一维张量,可以使用tensorflow的reduce_mean()函数来计算所有实例的交叉熵
		:param X:
		:return:
		'''
with tf.name_scope('loss'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	loss = tf.reduce_mean(xentropy, name='loss')
'''
函数sparse_softmax_cross_entropy_with_logits()与先应用softmax函数再计算交叉熵的效果是一样的
优势:高效,另外它还会处理一些边界如loits等于0的情况
'''
'''
现在我们有了神经网络模型,有了成本函数
下面定义一个梯度下降优化器(GradientDescentOptimizer),用来调整成本函数的值最小化
'''
learning_rate = 0.01
with tf.name_scope('train'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	traing_op = optimizer.minimize(loss)

'''
构建期的最后一个步骤:
制定如何对模型求值
我们简单地将精度作为性能指标
-----------------------
对于每个实例,通过检查最高logit值是否对应于目标类来确定神经网络的预测值是否正确,这里可以使用in_top_k()函数
这个函数会返回一个一维的张量,其值为布尔类型,因此我们需要将值强制装换成浮点型然后计算平均值,这会得出网络总体的精度
'''
with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

'''
创建结点初始化变量,创建Saver将训练后的模型保存到磁盘
'''
init = tf.global_variables_initializer()
saver = tf.train.Saver
