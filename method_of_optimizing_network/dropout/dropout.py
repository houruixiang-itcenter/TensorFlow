#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/19 下午3:31
# @Author  : Aries
# @Site    : 
# @File    : dropout.py
# @Software: PyCharm

'''
bagging适用于简单的模型

dropout适合复杂的神经网络
'''

import tensorflow as tf

'''
定义处理的数据
'''
x = tf.Variable(tf.ones([10, 10]))

'''
定义dro做为droput处理时的keep_prod参数
keep_prod:代表x中每个元素被保留下来得概率
'''
dro = tf.placeholder(tf.float32)

'''
定义一个dropout操作
函数    
def dropout(x, keep_prob=None, noise_shape=None, seed=None, name=None,rate=None)
仅仅关注 x---数据  keep_prob---x中每一个被保留下来的概率
'''
y = tf.nn.dropout(x, dro)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    '''
    数据中有一半变为0
           一半的数据 除以1/keep_prob
           
           
    ------------------------
    一般输入单元,会将keep_prob设置为0.8
    一般隐藏单元设置为0.5
    '''
    print(sess.run(y, feed_dict={dro: 0.5}))
