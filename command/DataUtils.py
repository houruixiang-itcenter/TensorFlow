#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/13 下午8:57
# @Author  : Aries
# @Site    : 
# @File    : DataUtils.py
# @Software: PyCharm
import os

from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# todo 全局存储序列化文件的位置
path = '/Users/houruixiang/python/Scikit-learn-master/command/assets'


def calRMSE(exceptVal, predictVal):
	'''
	:param exceptVal: --- 实际的期望值
	:param predictVal: ---- 当前模型的预测值
	:return:
	返回当前模型的均方根误差
	'''
	lin_mse = mean_squared_error(exceptVal, predictVal)
	lin_rmse = np.sqrt(lin_mse)
	return lin_rmse


def display_scores(scores):
	'''
	使用交叉验证的方式评估模型的性能
	:param scores:
	:return:
	'''
	print('Scores:', scores)
	print('Mean:', scores.mean())
	print('Standard deviation:', scores.std())


from sklearn.externals import joblib


def select_path(mode):
	if mode == 0:
		path = '/Users/houruixiang/python/TensorFlow/command/assets'
	return path


def serialize_data(tag_model, tag, mode=0):
	'''
	序列化data
	:param tag_model:序列化的对象
	:param tag:序列化文件存放位置
	:return:
	'''
	path = select_path(mode)
	tag = os.path.join(path, tag)
	joblib.dump(tag_model, tag)


def get_serialize_data(tag, mode=0):
	'''
	根据tag反序列化
	:param tag:
	:return:
	返回反序列化结果
	'''
	path = select_path(mode)
	tag = os.path.join(path, tag)
	return joblib.load(tag)
