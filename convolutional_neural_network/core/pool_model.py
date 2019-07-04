#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 下午8:35
# @Author  : Aries
# @Site    : 
# @File    : pool_model.py
# @Software: PyCharm

'''
池化层
'''
'''
池化层的作用:通过对输入图像进行二次采样以减小计算负载,内存利用率和参数数量(从而降低过度拟合的风险)

---------------------------
池化层仅仅是聚合数据:
1.必须定义接收野,步幅和填充类型
2.没有权重,不需要卷积数据,仅仅是聚合数据
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images

# load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape

'''
创建输入X的占位符和池化层
'''
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
'''
最大池化层:
每个池化内核,取最大的值进入下一层

--------
ksize:就是池化内核----  参数:batch_size & height & width & channels
目前tensorflow还不支持多个实例叠加 所以batch_size必须为  也不支持空间纬度和深度纬度的叠加,所以h and w必须同时等于1,或者channels必须等于1
'''
max_pool = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

with tf.Session() as sess:
	output = sess.run(max_pool, feed_dict={X: dataset})

plt.imshow(output[0].astype(np.uint8))
plt.show()
