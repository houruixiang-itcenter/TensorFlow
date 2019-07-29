#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/25 下午6:11
# @Author  : Aries
# @Site    :
# @File    : CNN_Cifar-10.py
# @Software: PyCharm
'''
CNN架构类
'''
import tensorflow as tf
import numpy as np
import time
import math

import CNN_master.CNN__CIFAR10_master.Cifar10_data as cifar

max_step = 4000
bath_size = 100
'''
评估数
'''
num_example_for_eval = 10000
data_dir = '/Users/houruixiang/python/TensorFlow/command/Cifar_data/cifar-10-batches'


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weights_loss')
        tf.add_to_collection('loss', weights_loss)

    return var


'''
获取训练数据和测试数据
训练数据 进行增强处理
测试数据 无需进行增强处理
'''
images_train, labels_train = cifar.input(data_dir=data_dir, batch_size=bath_size, distorted=True)
images_test, labels_test = cifar.input(data_dir=data_dir, batch_size=bath_size, distorted=None)

'''
创建x and  y_的占位符
'''
x = tf.placeholder(tf.float32, [bath_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [bath_size])

# todo 构建卷积神经网络的前馈网络  --- 卷积层 + 池化层
'''
卷积层1: conv1
-------------------------------------------------------------------------------------
构建第一个:
卷积层(使用relu做为激活函数) --- 有64个卷积核
接着使用最大池化层
'''
kernel1 = variable_with_weight_loss([5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))

'''
池化层1:pool1
--------------------------------------------------------------------------------------
池化:
池化核3*3 步长是2*2
'''
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

'''
卷积层2:conv2
---------------------------------------------------------------------------------------
卷积核[5,5,64,64]

'''
kernel2 = variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))

'''
池化层2:pool2
--------------------------------------------------------------------------------------
池化:
池化核3*3 步长是2*2
'''
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# todo 构建卷积神经网络的前馈网络  --- 两个卷积层之后的全连接层
'''
首先需要将一个卷积层(这里就是池化层pool2)拉直为一维数据
'''
reshape = tf.reshape(pool2, [bath_size, -1])
dim = reshape.get_shape()[1].value

'''
第一个全链接层:
1.隐藏单元是  384个 
2.w使用标准差为0.4的正态分布<这里定义为w1>
3.w1为了防止过度拟合使用L2正则优化参数
4.L2正则化参数是0.04
5.激活函数还是relu
'''
w1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, w1) + fc_bias1)

'''
第二个全链接层
隐藏单元 192个
'''
w2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, w2) + fc_bias2)

'''
第三全连接层是10个隐藏单元
'''
w3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.0, shape=[10]))
result = tf.add(tf.matmul(local4, w3), fc_bias3)

# todo 反向神经网络的调节
'''
这里使用sparse_softmax_cross_entropy_with_logits()函数计算损失值
每一层的value是有抽象的方法variable_with_weight_loss生成,这里会将你想要正则化的参数add_to_collection,
然后通过name可以get_conllection()获取损失值

这里我们使用的优化器是Adam算法来优化学习率
'''
# softmax的损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection('loss'))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

'''
计算准确率
'''
top_k_op = tf.nn.in_top_k(result, y_, 1)

# todo 进入会话执行阶段
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    '''
    开启多线程
    '''

    '''
    将下面一行注释掉再运行，发现程序不动了，这时处于一个挂起状态，start_queue_runners的作用是启动线程，向队列里面写数据。
    tf.train.start_queue_runners 这个函数将会启动输入管道的线程，填充样本到队列中，以便出队操作可以从队列中拿到样本。
    '''
    tf.train.start_queue_runners()

    for step in range(max_step):
        start_time = time.time()
        image_train, label_train = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_train, y_: label_train})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = bath_size / duration
            sec_per_batch = float(duration)

            '''
            打印每一轮训练的耗时
            '''
            print('step %d, loss = %.2f(%.1f examples/sec; %.3f sec/batch)' %
                  (step, loss_value, examples_per_sec, sec_per_batch))

    print('-------------------------------------------测试集来估计准确率-------------------------------------------------')
    true_count = 0
    num_batch = int(math.ceil(num_example_for_eval / bath_size))
    total_num_examples = num_batch * bath_size
    for i in range(num_batch):
        image_test, label_test = sess.run([images_test, labels_test])
        predict = sess.run([top_k_op], feed_dict={x: image_test, y_: label_test})
        true_count += np.sum(predict)

    print('accuracy = %.3f%%' % ((true_count / total_num_examples) * 100))


