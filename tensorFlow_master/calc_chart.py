#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/6 上午11:12
# @Author  : Aries
# @Site    : 
# @File    : calc_chart.py
# @Software: PyCharm
'''
创建一个计算图并在会话中执行
'''

print('-------------------------------------------------创建一个计算图并在会话中执行--------------------------------------')
import tensorflow as tf
import os
import _sqlite3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
下面的代码仅仅是创建计算机图
'''
x = tf.Variable(3, name='X')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2

x1 = tf.Variable(5, name='X')
y1 = tf.Variable(6, name='y')
f1 = x1 * x1 * y1 + y1 + 2
'''
到此为止,仅仅是创建了一个计算图,还并没有执行这个计算图

1.要执行这个图,需要打开一个tensorflow的会话,然后用他来初始化变量b并求值f
2.这个会话会将计算分发到诸如CPU或者GPU设备上并执行,他还持有所有变量的值


------------------------------------------------------------
下面的code是这样:
1.创建一个会话
2.初始化所有变量
3.求值
4.f关闭整个会话
'''
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

print('-------------------------------------------------with实现--------------------------------------')
'''
每次都要sess.run()看起来有些笨拙
我们可以使用with语法,而且会自动关闭session
'''
with tf.Session() as sess:
    x1.initializer.run()  # 等价于 tf.get_default_session().run(x1.initializer)
    y1.initializer.run()  # 等价于 tf.get_default_session().run(y1.initializer)
    result = f1.eval()  # tf.get_default_session().run(f)

print(result)

print('-------------------------------------------------全局初始化所有变量--------------------------------------')
'''
除了手工为每个变量调用初始化器之外还可以使用global_variables_initializer()函数来完成相同的操作

注意这个操作不会立刻初始化,他只是在图中创建了一个节点,这个节点会在会话执行时初始化所有变量"
'''
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
print(result)

print('-------------------------------------------------创建一个InteractiveSession--------------------------------------')
'''
创建一个InteractiveSession
他和常规会话的不同之处在于InteractiveSession在创建时会将子集设置为默认会话
因此你无须使用with块  但是在结尾处需要关闭会话
'''
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)

'''
一个tensorflow程序可以分为两部分:
1.第一部分用来构建一个计算图(称为构建阶段)----这个图用来展示ML模型和训练所需的计算
2.第二部分来执行这个图(称为执行阶段)----重复执行每一步训练动作(比如每个小批量执行一步),并逐步提升模型的参数
'''

print('-------------------------------------------------管理图--------------------------------------')
'''
你创建的所有节点都会自动添加到默认图上:
'''
x1 = tf.Variable(1)
'''
这样就构建一个默认的计算机图
'''
print(x1.graph is tf.get_default_graph())

'''
这其实是比较简单的,你利用tensorflow创建了一个计算机图,那么他就是tensorflow的默认图

但是有时候你回遇到这样的场景:
你可能想要管理多个互不依赖的图
解决方法:
创建一个新的图,然后通过with将其转化为一个默认图
'''
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())

print('-------------------------------------------------节点值得生命周期--------------------------------------')
'''
当求值一个节点时候,Tensorflow会自动检测该节点依赖的节点,并先对这些节点求值
'''
w = tf.constant(3)  # constant:不需要在运行中初始化值
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
    print(y.eval())
    print(z.eval())

'''
求值y:
Tensorflow会检测到y的值依赖于x,进而依赖于w,所以求y的步骤是这样的:
先求出w,然后求出x,最终求出y


求值z:
z依赖于x,进而依赖w,所以求值z的步骤:
先求出w,然后求出x,最终求出z


注意TensorFlow的求值步骤不会复用.所以w,x的计算会有两次


节点生命周期:每次执行图之前均会被丢弃
变量的值:会有会话来维护,所以,生命周期等于会话
'''

'''
就像上述的计算,两次计算均使用相同的w和x,但是却计算了两次,所以我们需要复用一切没必要重新计算的重复步骤
那么我们需要告诉TensorFlow在一次图中执行计算来获取y和z
'''
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)


'''
单进程的tensorflow:即使他们共享一个计算机图,各个会话之间还是要单独隔离互不影响的,每个会话会对每个变量有自己当前的拷贝

分布式TensorFlow:变量保存在每个服务器上,所以多个会话可以共享同一个变量

'''
