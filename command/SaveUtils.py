#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/11 下午5:56
# @Author  : Aries
# @Site    : 
# @File    : SaveUtils.py
# @Software: PyCharm
save_path = '/Users/houruixiang/python/TensorFlow/command/save_assets/'
import tensorflow as tf


def joint_params(path='theta.ckpt'):
	
	return save_path + path
	
	pass


def save(sess,
         save_path=save_path,
         global_step=None,
         latest_filename=None,
         meta_graph_suffix="meta",
         write_meta_graph=True,
         write_state=True,
         strip_default_attrs=False):
	saver = tf.train.Saver()
	path = joint_params(save_path)
	saver.save(sess,
	           path,
	           global_step,
	           latest_filename,
	           meta_graph_suffix,
	           write_meta_graph,
	           write_state,
	           strip_default_attrs)


def restore(sess, save_path=save_path):
	saver = tf.train.Saver()
	path = joint_params(save_path)
	saver.restore(sess, path)
