#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/16 上午12:45
# @Author  : Aries
# @Site    :
# @File    : dnn_execut.py
# @Software: PyCharm
'''
自定义DNN的执行类
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import artificial_neural_network.DNN.dnn_model as dnn
from command.SaveUtils import save, restore
from command.DataUtils import get_serialize_data
import numpy as np

mnist = get_serialize_data('mnist', 1)

'''
小批次梯度下降
定义epoch数量,以及小批次的大小
'''
n_epochs = 40
batch_size = 500

with tf.Session() as sess:
	dnn.init.run()
	for epoch in range(n_epochs):
		for iteration in range(mnist.train.num_examples // batch_size):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run(dnn.traing_op, feed_dict={dnn.X: X_batch, dnn.y: y_batch})
		acc_train = dnn.accuracy.eval(feed_dict={dnn.X: X_batch, dnn.y: y_batch})
		acc_test = dnn.accuracy.eval(feed_dict={dnn.X: mnist.test.images, dnn.y: mnist.test.labels})
		print(epoch, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)
	save_path = save(sess, './my_dnn_final.ckpt')
	
	'''
	上面代码先打开一个tensorflow的会话,运行初始化代码来初始化所有的变量
	1.运行主循环:在每一个周期中.迭代一组和训练集大小相对应的批次,每一个小批次通过next_batch()方法来获得
	2.执行训练操作,将当前小屁次的输入数据和目标传入
	3.接下来,在每个周期结束的时候,代码回用上一个小批次以及全量的训练集来评估模型,并打印结果,
	4.最后将模型的参数保存到硬盘
	'''

'''
使用神经网络
现在神经网络已经被训练好了,可以用它来做预测了,保留构建器的代码,修改执行的代码
'''

with tf.Session() as sess:
	restore(sess,'./my_dnn_final.ckpt')
	X_new_scaled = mnist.test.images
	Z = dnn.logits.eval(feed_dict={dnn.X: X_new_scaled})
	y_pred = np.argmax(Z, axis=1)
	print(y_pred)



