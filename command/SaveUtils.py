#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/11 下午5:56
# @Author  : Aries
# @Site    : 
# @File    : SaveUtils.py
# @Software: PyCharm
save_path = '/Users/houruixiang/python/TensorFlow/command/save_assets/'
import tensorflow as tf


def joint_params(mode):
    if mode == 0:
        return save_path + 'theta.ckpt'
    if mode == 1:
        return save_path + 'test.ckpt'

    pass


def save(sess,
         save_path=save_path,
         global_step=None,
         latest_filename=None,
         meta_graph_suffix="meta",
         write_meta_graph=True,
         write_state=True,
         strip_default_attrs=False, mode=0):
    saver = tf.train.Saver()
    path = joint_params(mode)
    saver.save(sess,
               path,
               global_step,
               latest_filename,
               meta_graph_suffix,
               write_meta_graph,
               write_state,
               strip_default_attrs)


def restore(sess, save_path=save_path, mode=0):
    saver = tf.train.Saver()
    path = joint_params(mode)
    saver.restore(sess, path)
