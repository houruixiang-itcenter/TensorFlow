#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/3 下午10:54
# @Author  : Aries
# @Site    :
# @File    : CNN_model.py
# @Software: PyCharm
'''
卷积神经网络
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images

# load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# create 2 filters 创建两个filters助手,也就是两个权重
'''
卷积神经可以共享权重和偏移值
再者相对于全连接:
1.卷积神经训练快
2.卷积神经可以在不同的位置下识别模式,而全连接层只能再固定位置下识别模式
'''
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
'''
1.构建中间是垂直白线的filter,即中间垂直白线是0
2.构建中间是水平白线的filter
'''
filters_test[:, 3, :, 0] = 1
filters_test[3, :, :, 1] = 1
filters = tf.reshape(filters_test, [7, 7, channels, 2])

'''
创建一个带有x和卷积层的图形，应用2个过滤器
'''
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
	output = sess.run(convolution, feed_dict={X: dataset})
	plt.imshow(output[0, :, :, 1])
	plt.show()
