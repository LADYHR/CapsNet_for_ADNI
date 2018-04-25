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
#  from utils import load_data
from CapsNet import CapsNet


def train(model, supervisor, num_label):



def main(_):
	tf.logging.info('Loading Graph...')
	num_label = 3
	model = CapsNet()
	tf.logging.info('Graph loaded!')
	sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)
	if cfg.is_training:
		tf.logging.info('Start Training...')
		train(model, sv, num_label)
		tf.logging.info('Train done!')
	else:
		evaluation(model, sv, num_label)
		tf.logging.info('Test done!')
	print("Main programming finished!")


if __name__ == "__main__":
	tf.app.run()


