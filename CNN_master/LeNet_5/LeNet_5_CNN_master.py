#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/29 下午5:38
# @Author  : Aries
# @Site    :
# @File    : LeNet_5_CNN_master.py
# @Software: PyCharm
'''
LeNet-5 for CNN  with MNIST
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers, learn

from command.DataUtils import get_serialize_data

# todo 准备数据
'''
获取mnist data
'''
mnist = get_serialize_data('mnist', 2)  # type: learn.datasets.base.Datasets

# todo 设置基础参数
batch_size = 100
learning_rate = 0.01
learning_rate_decay = 0.99
max_step = 30000


# todo 创建前馈神经网络
def hidden_layer(input_tensor, regularizer, avg_class, resuse):
    '''
    创建第一个卷积层,得到的特征图大小为32@28*28
    C1
    '''
    with tf.variable_scope('C1-conv', reuse=resuse):
        conv1_w = tf.get_variable('weight', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_b = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_w, strides=[1, 1, 1, 1], padding='SAME')
        '''
        激活函数 还是relu
        '''
        c1_result = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))

    '''
    创建第一个池化层  32@14*14
    '''
    with tf.variable_scope('S2-max_pool'):
        s1_pool = tf.nn.max_pool(c1_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    '''
    创建第二个卷积层,得到的特征图 --- C3  
    第一个池化层之后获得的输出是  32@28*28
    所以卷积核的深度应该是32
    又因为输出的层数是64 所以卷积核的个数是64
    '''
    with tf.variable_scope('C3-conv', reuse=resuse):
        conv3_w = tf.get_variable('weight', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_b = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(s1_pool, conv3_w, strides=[1, 1, 1, 1], padding='SAME')
        c3_result = tf.nn.relu(tf.nn.bias_add(conv3, conv3_b))

    '''
    创建第二个池化层,池化结果 64 @ 7*7
    '''
    with tf.variable_scope('S4-max_pool'):
        s4_pool = tf.nn.max_pool(c3_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        '''
        将最后一层池化层转化为全连接层
        -------------------------
        下面代码 shape[0] batch个数 shape[1]长度方向 shape[2]宽度方向 shape[3]深度方向
        '''
        shape = s4_pool.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]
        reshaped = tf.reshape(s4_pool, [shape[0], nodes])

    '''
    创建第一个全连接层 
    '''
    with tf.variable_scope('layer5-full', reuse=resuse):
        layer5_full_w = tf.get_variable('weight', [nodes, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
        '''
        对全连接层的w进行正则化处理
        '''
        tf.add_to_collection('loss', regularizer(layer5_full_w))
        layer5_full_b = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.1))
        '''
        是否使用滑动平均值
        '''
        if avg_class == None:
            full_5 = tf.nn.relu(tf.matmul(reshaped, layer5_full_w) + layer5_full_b)
        else:
            full_5 = tf.nn.relu(tf.matmul(reshaped, avg_class.average(layer5_full_w)) + avg_class.average(layer5_full_b))

    '''
    创建第二个 全连接层
    '''
    with tf.variable_scope('layer6-full', reuse=resuse):
        layer6_full_w = tf.get_variable('weight', [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        '''
        对全连接层的w进行正则化处理
        '''
        tf.add_to_collection('loss', regularizer(layer6_full_w))
        layer6_full_b = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.1))
        '''
        是否使用滑动平均值
        '''
        if avg_class == None:
            full_6 = tf.nn.relu(tf.matmul(full_5, layer6_full_w) + layer6_full_b)
        else:
            full_6 = tf.nn.relu(tf.matmul(full_5, avg_class.average(layer6_full_w)) + avg_class.average(layer6_full_b))
        return full_6


'''
定义 x,y  以及反向传播的相关参数
'''
x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[batch_size, 10])
'''
初始化正则函数
'''
regularizer = layers.l2_regularizer(0.001)

'''
对于使用滑动平均值 resuse设置为True  其他设置为false
'''
y = hidden_layer(x, regularizer, avg_class=None, resuse=False)
training_step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(0.99, training_step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())

'''
注意reuse为True在这里
'''
average_y = hidden_layer(x, regularizer, variable_averages, resuse=True)

'''
定义loss函数
'''
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('loss'))
'''
定义学习率  --- 指数衰减
'''
learning_rate = tf.train.exponential_decay(learning_rate, training_step,
                                           mnist.train.num_examples / batch_size, learning_rate_decay,
                                           staircase=True)
with tf.control_dependencies([training_step, variable_averages_op]):
    train_op = tf.no_op(name='train')

'''
定义评估参数
'''
crorent_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(crorent_prediction, tf.float32))

# todo 执行网络
'''
下面是执行网络的部分
'''

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(max_step):
        '''
        验证数据
        '''
        if i % 1000 == 0:
            x_val, y_val = mnist.validation.next_batch(batch_size)

            reshape_x2 = np.reshape(x_val, (batch_size, 28, 28, 1))

            variable_feed = {x: reshape_x2, y_: y_val}
            variable_accuracy = sess.run(accuracy, feed_dict=variable_feed)

            print('After %d training step(s), validation accuracy is %g%%' % (i, variable_accuracy * 100))

        '''
        训练数据
        '''
        x_train, y_train = mnist.train.next_batch(batch_size)
        reshape_xs = np.reshape(x_train, (batch_size, 28, 28, 1))
        sess.run(train_op, feed_dict={x: reshape_xs, y_: y_train})
