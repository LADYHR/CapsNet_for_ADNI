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
#!user/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from PIL import Image


rootpath = "H:\\ADNI_DATA\\TEST\\Axial_45"
'''
AD -- img1.png
      img2.png
      img3.png
      ...
MCI -- img1.png
       img2.png
       ...
NL -- img1.png
      img2.png
      ...
'''

classes = ['AD', 'MCI', "NL"]
writer = tf.python_io.TFRecordWriter("train.tfrecords")
for index, name in enumerate(classes, start=1):
	class_path = rootpath + name + "/"
	for img_path in os.listdir(class_path):
		img = Image.open(img_path)
		img_raw = img.tobytes()  # 图片转化为原生bytes
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),  # 定义数据格式中的标签
			"img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  # 定义数据格式中的数据
		}))
		writer.write(example.SerializeToString)  # 序列化为字符串
writer.close()



