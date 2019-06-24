#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/24 上午11:07
# @Author  : Aries
# @Site    : 
# @File    : learning_rate_optimization.py
# @Software: PyCharm

'''
学习效率调度
'''
'''
神经网络中,反向调节权重,会用到梯度下降
之前有提到一些防止梯度消失/爆炸,让函数更快收敛的方式:
1.良好的权重初始化(Xavier和He初始化)
2.选择良好的激活函数(ReLU,RReLU,ELU)
3.批量归一化
4.重用神经网络层(仅仅提高效率)
'''
'''
在梯度下降中:
不断的更新学习率要比固定学习率的下降高效很多
------------------------------------
1.预定分段常数学习效率
2.性能调度
3.指数雕刻度
4.功率调度 

详细见 ---> batch_normallization.py
'''