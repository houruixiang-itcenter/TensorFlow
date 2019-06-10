#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/10 下午10:50
# @Author  : Aries
# @Site    : 
# @File    : reg_gradient_descent.py
# @Software: PyCharm
'''
实现梯度下降
1.手工计算梯度
2.使用自动微分
'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from command.DataUtils import get_serialize_data
import tensorflow as tf

print('--------------------------------------------------手工计算梯度------------------------------------------------')
'''
1.函数random_uniform()会在图中创建一个节点,这个节点会生成一个张量,函数会根据传入的形状和值域来生成随机值来填充这个张量,类似Numpy的rand()
2.函数assgin()创建一个为变量赋值的节点,实现批量的梯度下降 ,即 theta(next step) = theta - 学习率 x 单位步数
3.主循环部分不断执行训练步骤(n_epochs次),每迭代100次,输出一次均方根误差,这个值会不断降低
'''

housing = get_serialize_data('housing')
housing_data_plus_bias = get_serialize_data('housing_data_plus_bias')

m, n = housing.data.shape
n_epochs = 1000
learning_rate = 0.01
pipeline = Pipeline([
	('std_scaler', StandardScaler())
])
scaled_housing_data_plus_bias = pipeline.fit_transform(housing_data_plus_bias)
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
# 获取初始的theta
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0),name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = 2 / m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)
'''
开启会话
'''
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			print('Epoch', epoch, 'MSE = ', mse.eval())
		sess.run(training_op)
	best_theta = theta.eval()
print('best_theta:  ', best_theta)

print('--------------------------------------------------使用自动微分------------------------------------------------')
'''
使用自动微分???
'''
print('--------------------------------------------------使用优化器------------------------------------------------')
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# # training_op = tf.assign(theta, theta - learning_rate * gradients)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

'''
使用11章的动态优化器
'''
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate)