#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/16 下午9:00
# @Author  : Aries
# @Site    : 
# @File    : dnn_notice.py
# @Software: PyCharm
'''
dnn的一些补充
'''
# 微调神经网络的超参数
'''
------------------------------------------------------------------------------------------------------------------------
神经网络的灵活性恰好是他的一个主要的短板:
有太多的超参数需要调整,不仅仅是可以使用任何的网络拓扑(神经元是如何彼此连接的),即便是简单的MLP,也有很多可以调整的参数:
#### 层数,每层神经元数,每层用的激活函数类型,初始化逻辑的权重
'''
# 隐藏层的个数
'''
------------------------------------------------------------------------------------------------------------------------
低级隐藏层:用以建模低层的结构(比如各种形状和方向的线段)
中级隐藏层:组合这些低层结构来建模中层结构(比如,正方形,圆形等)
高级隐藏层和输出层:组合这些中层结构来建模高层结构(比如,人脸)
'''
# 每个隐藏层的神经元的个数
'''
------------------------------------------------------------------------------------------------------------------------
输入和输出神经元个数:由任务输入输出类型决定,比如mnist需要784个输入神经元和10个输出神经元
隐藏层的神经元个数:这是一个黑参数,以前使用漏斗型来定义尺寸,每层神经元依次减少;当然你也可以将其变为相同个数
然后隐藏层的神经元个数作为超参数逐步去验证,直到过度拟合之前结束训练
'''
# 激活函数
'''
------------------------------------------------------------------------------------------------------------------------
隐藏层:ReLU,当然要根据具体情况决定
输出层:使用softmax输出概率,当然对于回归模型.可以不使用激活函数
'''