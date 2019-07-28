#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/19 下午5:43
# @Author  : Aries
# @Site    : 
# @File    : dnn_operation.py
# @Software: PyCharm
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python import learn

from command.DataUtils import get_serialize_data
import tensorflow as tf

'''
加载数据
'''
mnist = get_serialize_data('mnist', 2)  # type: learn.datasets.base.Datasets

# todo 第一步 定义网络中的相关参数 --- 反向调节用
batch_size = 100  # 设置小批量的size
learning_rate = 0.8  # 设置初始学习率
learning_rate_decay = 0.999  # 设置学习率的衰减
max_steps = 20000  # 最大训练步数

'''
定义训练轮数的变量 一般定义为不可训练的
'''
training_step = tf.Variable(0, dtype=tf.float32, trainable=True)
# todo 第二步 定义网络中的权重参数,偏置参数和前向传播过程
'''
1.定义网络层 hidden_layer
'''


def hidden_layer(input, w1, b1, w2, b2, layer_name):
    layer1 = tf.nn.relu(tf.matmul(input, w1) + b1)
    return tf.matmul(layer1, w2) + b2


'''
2.定义w,b
隐藏层 -- 500
'''
w1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[500]))

w2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))

'''
3. 定义x,y
'''
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, [None, 10], name='y-output')

'''
4.获取labels
'''
y = hidden_layer(x, w1, b1, w2, b2, 'y')

# todo 滑动平均值
'''
为了提高最终模型在测试数据上的表现,这里我们使用滑动平均值
'''
'''
1.初始化一个滑动平均类,衰减率是0.99
2.为了使模型在训练前期更新更快,这里提供num_updates是网络的训练轮数
'''
averages_class = tf.train.ExponentialMovingAverage(0.99, training_step)

'''
定义一个更新变量滑动平均值的操作需要向滑动平均类的apply()函数提供一个参数列表
train_variables()函数返回集合图上 Graph.TRAINABLE_VARIABLES中的元素,这个集合就是所有没有指定trainable_variables=False的参数
'''
averages_op = averages_class.apply(tf.trainable_variables())

'''
再次计算经过前馈网络预测的y值,这里使用了滑动平均,注意这里滑动平均仅仅是一个影子变量
'''
a =averages_class.average(w1)
average_y = hidden_layer(x, averages_class.average(w1), averages_class.average(b1),
                         averages_class.average(w2), averages_class.average(b2), 'average_y')

# todo 定义反向传播的参数
'''
定义loss函数
-----------
这里我们使用sqarse_softmax_cross_entropy_with_logits(_sential,labels,logdits,name)
与softmax_cross_entropy_with_logits()函数计算相同,但是会更适合每张地理类别的图,且只能支持一张图一种类别的场景
在tf 1.0.0中这个函数只能通过命名参数的方式来使用,在这里logits参数是神经网络不包括softmax层前向传播结果,labels参数给出了训练数据的正确答案
'''
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

'''
l2正则
'''
regularizer = layers.l2_regularizer(0.001)
regularization = regularizer(w1) + regularizer(w2)
loss = tf.reduce_mean(cross_entropy) + regularization

'''
用指数衰减法设置学习率,这里staircase采用默认的False,就是说学习率一直衰减

                      learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False,
                      name=None
'''
learning_rate = tf.train.exponential_decay(learning_rate, training_step, mnist.train.num_examples / batch_size,
                                           learning_rate_decay)

'''
使用优化器来优化交叉熵和正则化损失函数
'''
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)

'''
每次反向传播一次均要更新一次参数
'''
with tf.control_dependencies([training_step, averages_op]):
    train_op = tf.no_op(name='train')

# todo 定义准确率
'''
定义模型的预测准确率
argmax返回最大值的索引号
'''
crorent_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(crorent_prediction, tf.float32))

# todo 执行阶段
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    '''
    准备验证数据集
    '''
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    '''
    准备测试数据
    '''
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    '''
    runing...
    '''
    # for i in range(max_steps):
    #     if i % 1000 == 0:
    #         validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
    #         print('After %d steps,validate_accuracy is %g%%' % (i, validate_accuracy * 100))
    #
    for i in range(3):
        xs, ys = mnist.train.next_batch(batch_size=1)
        r1 = y.eval(feed_dict={x: xs})
        r2 = average_y.eval(feed_dict={x: xs})
        w = w1.eval()[1, :5]
        b = b1.eval()[:5]
        print(w)
        print(b)
        print(a.eval()[1,:5])
        print(r1)
        print(r2)
        xs, ys = mnist.train.next_batch(batch_size=100)
        sess.run(train_op, feed_dict={x: xs, y_: ys})
    # test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    # print('After %d steps,test_accuracy is %g%%' % (max_steps, test_accuracy * 100))
