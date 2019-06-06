#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/6 下午4:40
# @Author  : Aries
# @Site    : 
# @File    : tensorflow_reg.py
# @Software: PyCharm
'''
TensorFlow中线性回归
'''
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m,n = housing.shape

