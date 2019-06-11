#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/11 下午5:31
# @Author  : Aries
# @Site    : 
# @File    : saver.py
# @Software: PyCharm
'''
tensorflow 存取数据
'''
import tensorflow as tf

print('-----------------------------------------------保存和恢复模型Save-------------------------------------------------')
'''
一旦训练好了模型.就需要将模型的参数保存到硬盘上,这样就可以在任何时刻使用这些参数

另外你可能希望在训练过程中定期将检查点保存起来,这样电脑崩溃时,就可以从最近一个检查点恢复.而不是从头再来


在Tensorflow中,存取模型十分容易,在构造末期(在所有变量节点都创建之后),创建一个Saver节点,然后在执行期,调用save()方法,并传入一个会话和检查点文件
的路径即可保存模型

因为每个sess保存的不是节点所构成的图 而是
'''
import tensorflow as tf
import numpy as np
from command.SaveUtils import save, restore

x = tf.placeholder(tf.float32, shape=[None, 1])
y = 4 * x + 4

w = tf.Variable(tf.random_normal([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
y_predict = w * x + b

loss = tf.reduce_mean(tf.square(y - y_predict))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 用于验证
isTrain = False
# 用于训练
#isTrain = True
train_steps = 100
checkpoint_steps = 50

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    if isTrain:
        for i in range(train_steps):
            sess.run(train, feed_dict={x: x_data})
            if (i + 1) % checkpoint_steps == 0:
               save(sess,mode=1)
    else:
        restore(sess,mode=1)
        print(sess.run(w))
        print(sess.run(b))

    print(sess.run(w))
    print(sess.run(b))
