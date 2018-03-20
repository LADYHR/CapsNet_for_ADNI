'''
=====================================================
=====================================================
Copyright (c) 2018,LADYHR
All rights reserved

FileName: CapsNet.py
Abstract: This file defines the structure of the
Capsule Network.
=====================================================
=====================================================
'''
#!user/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from config import cfg
from utils import get_batch_data
from CapsLayer import CapsLayer

epsilon = 1e-9

class CapsNet(object):
	def __init__(self,is_training=True):
		self.graph = tf.Graph()
		with self.graph.as_default():
			if is_training:
				self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.numthreads)
				self.Y = tf.one_hot(self.label, depth=3, axis=1,dtype=tf.float32)

				self.build_arch()
				self.loss()
				self._summary()

				self.global_step = tf.Variable(0, name='global_step', trainable=False)
				self.optimizer = tf.train.AdamOptimizer()
				self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
			else:
				self.X = tf.placeholder(tf.float32,shape=(cfg.batch_size, 28, 28, 1))
				self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size,1))
				self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, 3,1))
				self.build_arch

		tf.logging.info('Setting up the main structure')

	def build_arch(self):
		with tf.variable_scope('Conv1_layer'):
			conv1 = tf.layers.conv2d()










