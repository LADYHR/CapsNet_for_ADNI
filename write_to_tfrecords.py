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
import os
from math import ceil
from PIL import Image


rootpath = "C:\\Users\\Hua Rui\\Git_Repository\\CapsNet_for_ADNI\\TEST\\Axial_45"
# rootpath = "C:\\Users\\Hua Rui\\Desktop\\digital"
'''
AD -- img1.png
      img2.png
      img3.png
      ...
MCI -- img1.pngc
       img2.png
       ...
NL -- img1.png
      img2.png
      ...
'''

classes = ['AD', 'MCI', 'NL']
# classes = ['1', '2', '3']
writer = tf.python_io.TFRecordWriter("train.tfrecords")  # 训练集数据
writer2 = tf.python_io.TFRecordWriter("valuation.tfrecords")  # 验证集数据
writer3 = tf.python_io.TFRecordWriter("test.tfrecords")  # 测试集数据
num_w1 = 0
num_w2 = 0
num_w3 = 0
for index, name in enumerate(classes):
    class_path = rootpath + '\\' + name
    file_num = len([x for x in os.listdir(class_path) ])
    i = 0
    for img_file in os.listdir(class_path):
        i += 1
        img_path = os.path.join(class_path, img_file)
        img = Image.open(img_path)
        # img.show()
        w, h = img.size
        # img = img.resize((50, 50))  # 对图像大小进行重采样
        # img.show()
        img_raw = img.tobytes()
        print(index, img_raw)
        if i < ceil(file_num * 0.6):
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),   # 定义数据格式中的标签
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))  # 定义数据格式中的数据
            writer.write(example.SerializeToString())  # 序列化为字符串
            num_w1 += 1

        if (i < ceil(file_num * 0.8)) and (i >= ceil(file_num * 0.6)):
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),   # 定义数据格式中的标签
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))  # 定义数据格式中的数据
            writer2.write(example.SerializeToString())  # 序列化为字符串
            num_w2 += 1

        if i >= ceil(file_num * 0.8):
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),   # 定义数据格式中的标签
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))  # 定义数据格式中的数据
            writer3.write(example.SerializeToString())  # 序列化为字符串
            num_w3 += 1
writer.close()
print("Train set has " + str(num_w1) + "data!")
writer2.close()
print("Valuation set has " + str(num_w2) + "data!")
writer3.close()
print("Test set has " + str(num_w3) + "data!")
print("Write TFRecords files done!")



