#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 下午3:49
# @Author  : Aries
# @Site    :
# @File    : batch_normallization.py
# @Software: PyCharm
'''
用tensorflow实现批量归一化
'''
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected

'''
在神经网络中,为了缓和梯度消失/爆炸,提高算法的收敛效率,要对数据进行零中心化和归一化操作

(X-均值)/标准差 进行零中心化和归一化的操作
'''
'''
方式一:
-------------
tensorflow中有提供一个方法
批量归一化的操作需要你自己计算均值和方差,然后做为参数传给这个方法,
而且你自己必须确定缩放和偏移

方式二:
-------------
tensorflow还有一个比较简单的方法,batch_normal函数,他提供了所有的参数,可以直接调用,
或者告诉fully_connected()函数去调用他
'''
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
bn_params = {
    'is_training': is_training,
    'decay': 0.99,
    'updates_collections': None
}

'''
1.fully_connected()来创建层
2.调用激活函数之前通过调用batch_norm()函数(通过参数bn_params)来进行输入归一化


注意:
默认情况下-----------------
batch_norm()只中心化,归一化和对输入进行偏移操作,但是并不缩放(缩放值恒为1);这样对于没有激活函数或者用ReLU激活函数的层
是有效果的,但是对于其他的激活函数,你需要设置"scale",即将bn_params设置为True
'''
hidden1 = fully_connected(X, n_hidden1, scope='hidden1',
                          normalizer_fn=batch_norm, normalizer_params=bn_params)
hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2',
                          normalizer_fn=batch_norm, normalizer_params=bn_params)
logits = fully_connected(hidden2, n_outputs, activation_fn=None, scope='outputs',
                         normalizer_fn=batch_norm, normalizer_params=bn_params)
