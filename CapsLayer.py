'''
=====================================================
=====================================================
Copyright (c) 2018,LADYHR
All rights reserved

FileName: CapsLayer.py
Abstract: This file is used to build a Capsule layer.
=====================================================
=====================================================
'''
#!user/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import  tensorflow as tf

from config import  cfg

epsilon = 1e-9

class CapsLayer(object): # class CapsLayer inherit class object
	def __init__(self, caps_num, num_outputs, vec_len, with_routing=True, layer_type='FC'):
		self.caps_num = caps_num
		self.num_outputs = num_outputs
		self.vec_len = vec_len
		self.with_routing = with_routing
		self.layer_type = layer_type

	def __call__(self, input, kernel_size=None, stride=None):
		if self.layer_type == 'CONV':
			self.kernel_size = kernel_size
			self.stride = stride

			if not self.with_routing:
				# assert input.get_shape() == [cfg.batch_size, 89, 89, 128]
				capsules = tf.contrib.layers.conv2d(input, self.num_outputs * vec_len,
													self.kernel_size, self.stride, padding='VALID',
													activation_fn=tf.nn.relu)
				capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))

				capsules = squash(capsules)
				# assert capsules.getshape() == [cfg.batch_size, self.caps_num, 8, 1]
				return(capsules)

		if self.layer_type == 'FC':
			if self.with_routing:
				self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))
				with tf.variable_scope('routing'):
					b_IJ = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
					capsules = routing(self.input, b_IJ)
					capsules = tf.squeeze(capsules, axis=1)

			return(capsules)

def routing(input, b_IJ):
	W = tf.get_variable()
