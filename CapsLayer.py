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
# !user/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from config import cfg

epsilon = 1e-9


class CapsLayer(object): # class CapsLayer inherit class object
	def __init__(self, caps_num_I, caps_num_J, num_outputs, vec_len_I,vec_len_J, with_routing=True, layer_type='FC'):
		self.caps_num_I = caps_num_I
		self.caps_num_J = caps_num_J
		self.num_outputs = num_outputs
		self.vec_len_I = vec_len_I
		self.vec_len_J = vec_len_J
		self.with_routing = with_routing
		self.layer_type = layer_type

	def __call__(self, input, kernel_size=None, stride=None):
		if self.layer_type == 'CONV':
			self.kernel_size = kernel_size
			self.stride = stride

			if not self.with_routing:
				# assert input.get_shape() == [cfg.batch_size, 89, 89, 128]
				capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len_J, self.kernel_size, self.stride, padding='VALID', activation_fn=tf.nn.relu)
				capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len_J, 1))
				capsules = squash(capsules)
				assert capsules.getshape() == [cfg.batch_size, self.caps_num_J, 8, 1]
				return capsules

		if self.layer_type == 'CAPS':
			if self.with_routing:
				self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))
				with tf.variable_scope('routing'):
					b_IJ = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
					capsules = routing(self.input, b_IJ)
					capsules = tf.squeeze(capsules, axis=1)
				return capsules

	def routing(self, input, b_IJ):
		W = tf.get_variable('Weight', shape=(1, self.caps_num_I, self.caps_num_J, self.vec_len_I, self.vec_len_J), dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=cfg.stddev))
		input = tf.tile(input,[1, 1, self.caps_num_J, 1, 1])
		W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1])
		assert input.get_shape() == [cfg.batch_size, self.caps_num_I, self.caps_num_J, self.vec_len_I, 1]
		u_hat = tf.matmul(W, input, transpose_a=True)  # last 2 dims mul
		assert u_hat.get_shape() == [cfg.batch_size, self.caps_num_I, self.caps_num_J, self.vec_len_J, 1]
		u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

		for r_iter in range(cfg.iter_routing):
			with tf.variable_scope('iter_'+str(r_iter)):
				c_IJ = tf.nn.softmax(b_IJ, dim=2)
				if r_iter == (cfg.iter_routing - 1):
					s_J = tf.multiply(c_IJ, u_hat)
					s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
					assert s_J.get_shape() == [cfg.batch_size, 1,self.caps_num_J, self.vec_len_J, 1]
					self.v_J = squash(s_J)
					assert self.v_J.get_shape() == [cfg.batch_size, 1, self.caps_num_J, self.vec_len_J, 1]

				elif r_iter < (cfg.iter_routing - 1):
					s_J = tf.multiply(c_IJ, u_hat_stopped)
					s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
					self.v_J = squash(s_J)
					v_J_tiled = tf.tile(self.v_J, [1, self.caps_num_J, 1, 1, 1])
					assert v_J_tiled.get_shape() == [cfg.batch_size, self.caps_num_I, self.caps_num_J, self.vec_len_J, 1]
					u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
					assert u_produce_v.get_shape == [cfg.batch_size, self.caps_num_I, self.caps_num_I, 1, 1]
					b_IJ += u_produce_v
		return self.v_J


def squash(vector):
	vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
	scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
	vec_squashed = scalar_factor * vector
	return vec_squashed


