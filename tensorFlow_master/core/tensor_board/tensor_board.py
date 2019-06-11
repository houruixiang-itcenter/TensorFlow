#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/11 下午7:43
# @Author  : Aries
# @Site    : 
# @File    : tensor_board.py
# @Software: PyCharm
'''
用TensorBoard来可视化图和训练曲线
'''
import tensorflow as tf
import numpy as np
from command.DataUtils import get_serialize_data
from datetime import datetime

'''
使用tensorBoard可以基于一些训练状态,在浏览器上将这些状态以交互的方式展示出来,还可以将学习曲线将这些状态以交互方式展示出来
这种方式对识别图中的错误,发现图的瓶颈等非常有用


---------------------------------------
首先对图稍做修改,对算法状态以交互方式绘制图

为了方式值日文件合并我们使用时间戳来命名日志文件夹,防止状态文件合并
'''
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

'''
----------------------------------------------------------------------------------------------------------------------
下面是之前求小批量梯度的代码
'''
housing = get_serialize_data('housing')
m, n = housing.data.shape
scaled_housing_data_plus_bias = get_serialize_data('scaled_housing_data_plus_bias')

X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
n_epochs = 10
learning_rate = 0.01
batch_size = 100
n_batches = np.int(np.ceil(m / batch_size))

X_train = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y_train = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch


# 获取初始的theta
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
'''
命名作用域
在处理注入神经网络等复杂模型时,图很容易变得杂乱而庞大
为了避免这种情况,可以创建命名作用域来将相关的节点分组


eg:将error  和mse 定义到一个叫做'loss'的命名作用域中:
'''
with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
print(error.op.name)
print(mse.op.name)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate)  # 替代 gradients = 2 / m * tf.matmul(tf.transpose(X), error)
training_op = optimizer.minimize(mse)  # 替代 training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()

'''
初始化计算节点
'''
mse_summary = tf.summary.scalar('MSE', mse)  # 这个节点用来求MSE的值,并将其写入与TensorBoard称为汇总的二进制日志字符串中
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())  # 创建了一个用来将汇总写入到日志目录的FileWriter

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str,step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()
file_writer.close()