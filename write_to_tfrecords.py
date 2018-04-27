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
from PIL import Image


# rootpath = "C:\\Users\\LADYHR\\Git_Repository\\CapsNet_for_ADNI\\TEST\\Axial_45"
rootpath = "C:\\Users\\Hua Rui\\Git_Repository\\CapsNet_for_ADNI\\TEST\\Axial_45"
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

classes = ['AD', 'MCI', 'NL']
writer = tf.python_io.TFRecordWriter("train.tfrecords")
writer2 = tf.python_io.TFRecordWriter("valuation.tfrecords")
writer3 = tf.python_io.TFRecordWriter("test.tfrecords")
for index, name in enumerate(classes, start=1):
    class_path = rootpath + '\\' + name
    file_num = len([x for x in os.listdir(class_path) if os.path.isfile(x)])
    i = 0
    for img_file in os.listdir(class_path):
        i += 1
        img_path = os.path.join(class_path, img_file)
        img = Image.open(img_path)
        w, h = img.size
        img = img.resize((91, 109))
        # img.show()
        img_raw = img.tobytes()
        print(index, img_raw)
        if i < file_num * 0.6:
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),   # 定义数据格式中的标签
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))  # 定义数据格式中的数据
            writer.write(example.SerializeToString())  # 序列化为字符串
writer.close()
print("Finished!")



