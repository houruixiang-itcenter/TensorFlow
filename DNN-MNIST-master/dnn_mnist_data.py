#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/19 下午4:42
# @Author  : Aries
# @Site    :
# @File    : dnn_mnist_data.py
# @Software: PyCharm
'''
MNIST DNN神经网络构建
'''
import os

from tensorflow.contrib.learn.python import learn
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from command.DataUtils import serialize_data, get_serialize_data

'''
1. 加载MNIST数据集
'''
# mnist = input_data.read_data_sets('/Users/houruixiang/python/TensorFlow/command/assets_mnist', one_hot=True)
# serialize_data(mnist, 'mnist', 2)
mnist = get_serialize_data('mnist', 2)  # type: learn.datasets.base.Datasets

print('Training data and label size: ')
print(mnist.train.images.shape, mnist.train.labels.shape)
print('Testing data and label size: ')
print(mnist.test.images.shape, mnist.test.labels.shape)
print('Validation data and label size: ')
print(mnist.validation.images.shape, mnist.validation.labels.shape)

print('Example training data: ', mnist.train.images[0])
print('Example training label: ', mnist.train.labels[0])


