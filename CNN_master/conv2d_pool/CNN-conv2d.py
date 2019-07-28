#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/24 下午5:32
# @Author  : Aries
# @Site    : 
# @File    : CNN-conv2d_pool.py
# @Software: PyCharm
'''
实现一个简单的卷积层
'''
import tensorflow as tf
import numpy as np

# todo 设置数据
'''
设置输入格式的矩阵
'''
M = np.array([[[2], [1], [2], [-1]], [[0], [-1], [3], [0]], [[2], [1], [-1], [4]]],
             dtype='float32').reshape(1, 4, 3, 1)

# todo 设置必要参数
'''
创建cnn的filer,这里声明一个四维的矩阵
为了复用变量
'''
weights = tf.get_variable('weights', [2, 2, 1, 1], initializer=tf.constant_initializer([[-1, 4], [2, 1]]))
'''
创建偏置项
'''
biase = tf.get_variable('biase', [1], initializer=tf.constant_initializer(1))

x = tf.placeholder('float32', [1, None, None, 1])

# todo 设置卷积层的前馈
'''
设置卷积层
params:
1.input [batch_index,w,h,deep]
2.filer [w,h,input-deep,deep]
3.第一个 and 第四个 默认是1  中间代表宽和长方向上的步长
4.padding -- 'VALID'默认不补0,'SAME'边缘补0

SAME : 自动补0 (右边和下边)
'''
conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME')

'''
卷积层+偏置项
'''
add_bias = tf.nn.bias_add(conv, biase)

'''
设置池化层
这里使用最大池化层
'''
pool = tf.nn.max_pool(add_bias, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

# todo exect
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    conv_result = sess.run(add_bias, feed_dict={x: M})
    pool_result = sess.run(pool, feed_dict={x: M})

print(M)
print('------------------after conv_2d----------------------')
print(conv_result)
print('------------------after max_pool----------------------')
print(pool_result)