'''
=====================================================
=====================================================
Copyright (c) 2018,LADYHR
All rights reserved

FileName: main.py
Abstract: This is a main program. Aimed at using
Capsule Network to design a classifier for AD/MCI/NC.
=====================================================
=====================================================
'''
#!user/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import cfg
from utils import load_data
from CapsNet import CapsNet

def main():
	tf.logging.info('Loading Graph...')
	label_num = 3
	model = CapsNet()
	tf.logging.info('Graph loaded!')


if __name__ == "__main__":
	tf.app.run()

