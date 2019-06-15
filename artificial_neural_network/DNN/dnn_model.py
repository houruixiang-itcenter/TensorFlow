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


class MyDNNClassifier:
    '''
    构建阶段
    首先需要引入tensorflow库,然后是指定输入和输出的个数,并设置每层隐藏神经元的个数
    '''
    import tensorflow as tf
    import numpy as np

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    '''
    紧接着定义X,Y的占位符
    X: 用作输入层
    '''
    X = tf.placeholder(tf.float64, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')

    def __init__(self) -> None:
        super().__init__()

    '''
    接下来我们来创建神经网络
    1.占位符节点X将用作输入层;在执行期,它每次都会被训练批次替换(注意训练批次中的所有实例将由神经网络同时处理)
    2.创建两个隐藏层;两个隐藏层基本上是一样的:唯一的区别是他们和谁链接,以及每层中包含的神经元数量
    3.创建一个输出层:和隐藏层不同,它会用softmax而不是ReLU做为激活函数
    4.我们创建一个neuron_layer()函数来每次创建一个层;
    ----- 它需要的参数包括:输入,神经元数量,激活函数,层次的名字:
    '''

    def neuron_layer(self, X, n_neurons, name, activation=None):
        tf = self.tf
        np = self.np
        with tf.name_scope(name):
            # 输出数据量的number
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name='weights')
            b = tf.Variable(tf.zeros([n_neurons]), name='biases')
            z = tf.matmul(X,W) + b
            if activation == 'relu':
                return tf.nn.relu(z)
            else:
                return z
