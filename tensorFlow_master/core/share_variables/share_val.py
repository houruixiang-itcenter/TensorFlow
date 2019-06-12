#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/12 下午11:00
# @Author  : Aries
# @Site    :
# @File    : share_val.py
# @Software: PyCharm
'''
共享变量
'''
import tensorflow as tf

'''
tensorflow中模块化之后 就产生了一个问题如何在组件中共享变量?

----------------------------
'''
print('------------------------------------------方式一:每次调用需要给函数传递参数:----------------------------------------')


def relu(X, threshold):
	with tf.name_scope('relu'):
		w_shape = (int(X.get_shape()[1]), 1)
		w = tf.Variable(tf.random_normal((w_shape)), name='weights')
		b = tf.Variable(0.0, name='bias')
		z = tf.add(tf.matmul(X, w), b, name='z')
		return tf.maximum(z, threshold, name='relu')


n_features = 3
threshold = tf.Variable(0.0, name='threshold')
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name='output')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(len(relus)):
		result = relus[i].eval(feed_dict={X: [[1, 2, 3]]})
		print(result)

print('------------------------------------------方式二:第一次调用时候初始化变量------------------------------------------')
print('------------------------------------------方式三:tensorflow-api提供------------------------------------------')
'''
如果共享变量不存在,该方法先通过get_variable()函数创建共享变量,如果已经存在就复用这个变量;
期望的行为是通过当前的variable_scope()的一个属性来控制(创建或者复用)

eg:下面代码回创建一个名为'relu/threshold'的变量
注意:设定shape(),所以结果是一个标量
'''
with tf.variable_scope('relu'):
	threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))
'''
注意,如果这个遍历之前已经被get_variable()调用创建过,这里弧抛出一个异常
这种机制是为了避免由于误操作而复用变量;
如果要复用一个变量,需要通过设置变量的作用域的reuse为True来先显示实现(在这里不必制定形状或者初始化器)
'''
with tf.variable_scope('relu', reuse=True):
	threshold = tf.get_variable('threshold')

'''
上面这段代码回获取既有的'relu/threshold'变量,如果该变量不存在,或者再调用get_variable()时没有创建成功,则会抛出异常
下面来看一个和上面代码等价的实现
'''
with tf.variable_scope('relu') as scope:
	scope.reuse_variables()  # 等价于再构造中设置  reuse=True
	threshold = tf.get_variable('threshold')

'''
下面来看具体的实现
'''


def relu(X):
	with tf.variable_scope('relu', reuse=True):
		threshold = tf.get_variable('threshold')
		
		w_shape = (int(X.get_shape()[1]), 1)
		w = tf.Variable(tf.random_normal((w_shape)), name='weights')
		b = tf.Variable(0.0, name='bias')
		z = tf.add(tf.matmul(X, w), b, name='z')
		return tf.maximum(z, threshold, name='relu')


n_features = 3
# todo 创建变量
with tf.variable_scope('relu'):
	threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name='output')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(len(relus)):
		result = relus[i].eval(feed_dict={X: [[1, 2, 3]]})
		print(result)

'''
比较尴尬的一点 这个变量的初始化必须在函数之外

要解决这个问题:
只需要第一次reuse设为False进行初始化

之后reuse设为True进行复用
'''


def relu(X):
	threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))
	
	w_shape = (int(X.get_shape()[1]), 1)
	w = tf.Variable(tf.random_normal((w_shape)), name='weights')
	b = tf.Variable(0.0, name='bias')
	z = tf.add(tf.matmul(X, w), b, name='z')
	return tf.maximum(z, threshold, name='relu')


n_features = 3
relus = []
for relu_index in range(5):
	with tf.variable_scope('relu', reuse=(relu_index >= 1)) as scope:
		relus.appand(relu(X))
output = tf.add_n(relus, name='output')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(len(relus)):
		result = relus[i].eval(feed_dict={X: [[1, 2, 3]]})
		print(result)
