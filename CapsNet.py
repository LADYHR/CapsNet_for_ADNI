'''
=====================================================
=====================================================
Copyright (c) 2018,LADYHR
All rights reserved

FileName: CapsNet.py
Abstract: This file defines the structure of the
Capsule Network. And add summary to TensorBoard.
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
				self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
				self.Y = tf.one_hot(self.labels, depth=3, axis=1, dtype=tf.float32)

				self.build_arch()
				self.loss()
				self._summary()

				self.global_step = tf.Variable(0, name='global_step', trainable=False)
				self.optimizer = tf.train.AdamOptimizer()
				self.train_op = self.optimizer.minimize(self.margin_loss, global_step=self.global_step)
			else:
				self.X = tf.placeholder(tf.float32,shape=(cfg.batch_size, 91, 91, 1))
				self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size,1))
				self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, 3,1))
				self.build_arch()

		tf.logging.info('Setting up the main structure')

	def build_arch(self):
		with tf.variable_scope('Conv1_layer'):
			#  conv1, [batch_size, 91, 91, 100]
			conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=128, kernel_size=3, stride=1, padding='VALID')
			assert conv1.get_shape() == [cfg.batch_size, 89, 107, 128]

		with tf.variable_scope('PrimaryCaps_layer'):
			#  Primary Capsule layer, [batch_size, 41*41*16,8,1]
			primaryCaps = CapsLayer(caps_num_I=0, caps_num_J=32800, num_outputs=16,vec_len_I=1, vec_len_J=8, with_routing=False, layer_type='CONV')
			caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
			assert caps1.get_shape() == [cfg.batch_size, 32800, 8, 1]
			#  Primary Capsule layer2, [batch_size, 17*17*16,8,1]
			primaryCaps2 = CapsLayer(caps_num_I=32800, caps_num_J=16, num_outputs=16, vec_len_I=8, vec_len_J=8, with_routing=True, layer_type='CAPS')
			caps2 = primaryCaps2(caps1)
			assert caps2.get_shape() == [cfg.batch_size, 16, 8, 1]

		with tf.variable_scope('DigitCaps_layer'):
			digitCaps = CapsLayer(caps_num_I=16, caps_num_J=3, num_outputs=3, vec_len_I=8, vec_len_J=16, with_routing=True, layer_type='CAPS')
			self.caps3 = digitCaps(caps2)
			# assert caps3.get_shape() == [cfg.batch_size, 3, 16, 1]???

	def loss(self):
		#  calculate ||v_c||, then do softmax and calculate the predicted labels
		self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps3), axis=2, keep_dims=True)+epsilon)  # axis=2 ???
		self.softmax_v = tf.nn.softmax(self.v_length,dim=1)
		assert self.softmax_v.get_shape() == [cfg.batch_size, 3, 1, 1]
		self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v,axis=1))
		assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
		self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size,))

		#  calculate the margin loss
		max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
		max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
		assert max_l.get_shape() == [cfg.batch_size, 3, 1, 1]
		max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
		max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))
		T_c = self.Y
		L_c = T_c * max_l + cfg.lambdal * (1-T_c) * max_r
		self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

	def _summary(self):  # start with '_' means this func is a internal func
		train_summary = []
		train_summary.append(tf.summary.scalar('train/margin_loss',self.margin_loss))
		self.train_summary = tf.summary.merge(train_summary)

		correct_prediction = tf.equal(tf.to_int32(self.labels),self.argmax_idx)
		self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))











