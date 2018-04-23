'''
=====================================================
=====================================================
Copyright (c) 2018,LADYHR
All rights reserved

FileName: utils.py
Abstract: This file is used to load data and get
batch data.
=====================================================
=====================================================
'''
# !user/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

def read_and_decode(filename):
	# 根据文件名生成一个文件队列
	filename_queue = tf.train.string_input_producer([filename])
	#  根据文件格式选择对应的文件阅读器，创建reader
	reader = tf.TFRecordReader()
	# 从文件队列中读取一个序列化的样本
	key, serialized_example = reader.read(filename_queue)
	#  解析序列化的样本，提取特征
	features = tf.parse_single_example(serialized_example,
	                                   features={
		                                   'label': tf.FixedLenFeature([], tf.int64),
		                                   'img_raw': tf.FixedLenFeature([], tf.string),
	                                   })
	img = tf.decode_raw(features['img_raw'], tf.uint8)
	img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
	label = tf.cast(features['label'], tf.int32)

	return img, label

def get_batch_data(dataset, batch_size, num_threads):
	pass

if __name__ == "__main__":
	img, label = read_and_decode("train.tfrecords")
	print(img, label)
	print("Finished!")