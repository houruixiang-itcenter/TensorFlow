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
    if mode == 1:
        path = '/Users/houruixiang/python/TensorFlow/command/assets_ann'
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


'''
------------------------------------------------------------------------------------------------------------------------
获取mnist
'''


def get_mnist_data_and_target():
    mnist = get_serialize_data('mnist')
    x, y = mnist['data'], mnist['target']
    return x, y


def get_train_and_test():
    X, Y = get_mnist_data_and_target()
    X_train, X_test, Y_train, Y_test = X[:60000], X[10000:], Y[:60000], Y[10000:]
    shuffle_train_index = np.random.permutation(60000)
    shuffle_test_index = np.random.permutation(10000)
    try:
        if get_serialize_data('X_train') is None and get_serialize_data('Y_train') is None:
            print('x is none')
            X_train, Y_train = X_train[shuffle_train_index], Y_train[shuffle_train_index]
            serialize_data(X_train, 'X_train')
            serialize_data(Y_train, 'Y_train')
        else:
            print('x is not none')
            X_train, Y_train = get_serialize_data('X_train'), get_serialize_data('Y_train')

        if get_serialize_data('X_test') is None and get_serialize_data('Y_test'):
            print('y is none')
            X_test, Y_test = X_test[shuffle_test_index], Y_test[shuffle_test_index]
            serialize_data(X_test, 'X_test')
            serialize_data(Y_test, 'Y_test')
        else:
            print('y is not none')
            X_test, Y_test = get_serialize_data('X_test'), get_serialize_data('Y_test')
    except FileNotFoundError as e:
        X_train, Y_train = X_train[shuffle_train_index], Y_train[shuffle_train_index]
        X_test, Y_test = X_test[shuffle_test_index], Y_test[shuffle_test_index]
        serialize_data(X_train, 'X_train')
        serialize_data(Y_train, 'Y_train')
        serialize_data(X_test, 'X_test')
        serialize_data(Y_test, 'Y_test')

    return X_train, Y_train, X_test, Y_test


def get_mnist_train():
    if get_serialize_data('X_train') is None and get_serialize_data('Y_train') is None:
        X_train, Y_train, X_test, Y_test = get_train_and_test()
    else:
        X_train, Y_train = get_serialize_data('X_train'), get_serialize_data('Y_train')
    return X_train, Y_train


def get_mnist_test():
    if get_serialize_data('X_test') is None and get_serialize_data('Y_test') is None:
        X_train, Y_train, X_test, Y_test = get_train_and_test()
    else:
        X_test, Y_test = get_serialize_data('X_test'), get_serialize_data('Y_test')
    return X_test, Y_test
