# !/usr/bin/python
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

from command.DataUtils import get_serialize_data,serialize_data
import tensorflow as tf
import numpy as np
from command.SaveUtils import save, restore

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
serialize_data(scaled_housing_data_plus_bias,'scaled_housing_data_plus_bias')
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
# 获取初始的theta
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
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
        # training_op.eval()
    best_theta = theta.eval()
print('best_theta:  ', best_theta)

print('--------------------------------------------------使用自动微分------------------------------------------------')
'''
使用自动微分???
'''
print('--------------------------------------------------使用优化器------------------------------------------------')
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate)  # 替代 gradients = 2 / m * tf.matmul(tf.transpose(X), error)
training_op = optimizer.minimize(mse)  # 替代 training_op = tf.assign(theta, theta - learning_rate * gradients)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch', epoch, 'MSE = ', mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
print('best_theta:  ', best_theta)
'''
使用11章的动态优化器
'''
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
print('-------------------------------------给算法提供训练数据---小批次梯度下降--------------------------------------------')
'''
把上面代码使用小批次梯度下降来实现
需要每次迭代时用下一个小批量替换X和y

-----------------------------
所以我们需要一个占位符的节点,他比较特殊,不会进行任何实际的计算,而是运行时输出其想输出的值
一般他用来训练过程中,将值传给tensorflow,如果运行时不指定初始值会抛出异常

就是说占位符节点不需要在运行时候声明变量
'''
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
print(B_val_1)
print(B_val_2)

def fetch_batch(epoch, batch_index, batch_size):
    '''
    load the data from disk
    '''
    # 随机获取小批次数据
    '''
    seed 决定生成随机数是否一致
    '''
    np.random.seed(epoch * n_batches + batch_index)
    '''
    从实例中随机生成100组实例
    '''
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch
'''
实际上,可以输入任何操作的输出,不仅仅是占位符
下面来看下小批次梯度下降的code:
'''
'''
 定义批次的大小并计算批次的总数
np.ceil:向上取整
'''
'''
最后在执行阶段,逐个获取小批次,然后在评估依赖于他们的节点时候,通过feed_dict参数提供X,y的值
'''
X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
n_epochs = 10
batch_size = 100
n_batches = np.int(np.ceil(m / batch_size))
X_train = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y_train = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')


# 获取初始的theta
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate)  # 替代 gradients = 2 / m * tf.matmul(tf.transpose(X), error)
training_op = optimizer.minimize(mse)  # 替代 training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()
    save(sess,mode=0)
print(best_theta)
