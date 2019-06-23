#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/23 下午5:25
# @Author  : Aries
# @Site    : 
# @File    : tf_saver.py
# @Software: PyCharm
'''
saver
'''
import tensorflow as tf
from command.SaveUtils import restore, save

print('-------------------------------------------------重用tensorflow模型---------------------------------------------')
'''
如果原有的模型是tensorflow训练的,你可以轻松地还原该模型并在新任务中接着训练它
'''
# with tf.Session() as sess:
# 	restore(sess, './batch_normallization/final_model')
# 	'''
# 	Train it on your new Task...
# 	'''

'''
通常我们只会还原原有模型的一部分,下面我们还原:
隐藏层1 & 2
'''

'''
1.创建一个新模型,确保它复制了原有模型隐藏层的目标层
2.创建一个节点来初始化所有参数
3.获取一个参数列表,并使用hidden[12]这个正则参数来做筛选
4.创建一个dict来存放原有模型和新模型里面名字的映射(通常不变)
5.分别创建一个original的saver来还原旧数据和一个新的saver来存储新数据
6.开启会话,初始化新模型中的所有参数,然后将层1-2的参数从已有模型中进行重建
7.最后,在新任务中训练新模型,并存储他
'''
# build a new model with the same definition as before for hidden layers
init = tf.global_variables_initializer()

reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='hidden[12]')
reuse_vars_dict = dict([(var.name, var.name) for var in reuse_vars])
original_saver = tf.train.Saver(reuse_vars_dict)

with tf.Session() as sess:
	sess.run(init)
	restore(sess, './batch_normallization/final_model', saver=original_saver)
	# tarin the new model
	save(sess, './batch_new/final_model')

print('-------------------------------------------------重用其他框架的模型---------------------------------------------')
'''
使用其他框架时候,需要不断手动加载所有的权重,然后将他们赋给适当的参数
其实不是很懂,稍候补充
'''
print('-------------------------------------------------冻结底层---------------------------------------------')
'''
第一个DNN 的底层肯能已经学会了检测图像中dxszxxzxxzz        ccccccccccccccccccccccccccc的低级特征,这对于两个图像分类来说是有用的
训练新的DNN时,通常来讲"冻结"权重是一个比较好的方法:如果底层权重被固定,那么高层的权重就比较容易训练
为了在训练中冻结底层,最简单的方法就是给优化器列出姚训练的变量列别,除了底层的变量
code1:获得所在隐藏层3-4的可训练变量,排除了在隐藏层1和隐藏层2的变量;
code2:我们把这个可训练变量的受限列表传给优化器的minize()函数


现在层1和层2被冻结了:他在训练时候不在抖动(通常被称为冻结层)
'''
# train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='hidden[3,4]outputs')
# training_op = optimizer.minimize(loss,var_list=train_vars)

print('-------------------------------------------------缓存冻结层---------------------------------------------')
'''
由于当前我们是基于DNN的两层作为基础层,所以
我们可以把前两层看做是预测器,所以通过X_train拿到所有的最高冻结层的输出
然后将这个作为特征值进行模型训练
'''
print('------------------------------------------------调整,丢弃或者替换高层---------------------------------------------')
'''
其实就是基于其他神经网络做一些调整,丢弃或者替换高层
一般是逐步解冻,以至于获得更好的神经网络层
'''

print('------------------------------------------------模拟动物园---------------------------------------------')
