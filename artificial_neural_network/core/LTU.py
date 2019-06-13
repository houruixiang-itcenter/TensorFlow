#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/13 下午10:35
# @Author  : Aries
# @Site    : 
# @File    : LTU.py
# @Software: PyCharm
'''
人工神经网络: ANN
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

'''
输入:输入神经网络
每一个输出:代表一个神经网络  eg:入鸢尾花有3个输出神经网络

感知器是最简单的ANN架构之一,它基于一个稍微不同的被称为线性阈值单元(LTU)的人工神经元:
1.输入输出都是数字,每个输入的连接都有一个对应的权重

2.LTU会加权求和(z = wT * X)

3.对求值结果应用一个阶跃函数.最后求得最后的输出:  step(z)


基于鸢尾花,看下面的代码:
scikit代码 提供了一个实现单一LTU网络的Perceptron类,他基本可以再鸢尾花的数据集上如期工作:
'''

iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int) # is Setosa?
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])
print(y_pred)


'''
注意和逻辑回归分类器相反,感知器不成熟出某个概率,它只是更具一个固定的阈值来做预测(也就是说智慧输出0 或者 1这样)
所以可以直接看出单个感知器是不如逻辑分类的
所以我们要将多个感知器堆叠起来进行弥补这个弊端.这种形式的ANN叫做多层感知器[MLP]
MLP同样可以解决异或者同的问题xxxx
'''
