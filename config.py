'''
=====================================================
=====================================================
Copyright (c) 2018,LADYHR
All rights reserved

FileName: config.py
Abstract: This file sets some parameters.
=====================================================
=====================================================
'''
#!user/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

flags = tf.app.flags  # add optional argument in command line
cfg = tf.app.flags.FLAGS  # get corresponding argument

############################
#	 hyper parameters      #
############################

# for separate margin loss
flags.DEFINE_float('m_plus',0.9,'the parameter of m_plus')
flags.DEFINE_float('m_minus',0.1,'the parameter of m_minus')
flags.DEFINE_float('lambdal',0.5,'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size',64,'batch_size')
flags.DEFINE_integer('epoch',5,'epoch')
flags.DEFINE_integer('routing_iter_num',3,'number of iterations in routing algorithm')
flags.DEFINE_float('stddev',0.01,'stddev for W initializer')


############################
#  environment setting     #
############################
flags.DEFINE_string('dataset','ADNI1_1.5T','the name of dataset [ADNI1_1.5T]')
flags.DEFINE_boolean('is_training',True,'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
flags.DEFINE_string('logdir','logdir','logs directory')
flags.DEFINE_integer('train_summary_freq',1,'the frequency of saving train summary(step)')
flags.DEFINE_integer('test_summary_freq',2,'the frequency of saving test summary(step)')
flags.DEFINE_integer('save_model_freq',3,'the frequency of saving model(epoch)')
flags.DEFINE_string('results_path','results','path for saving results')



############################
#  distributed setting     #
############################
flags.DEFINE_integer('num_gpu',2,'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu',64,'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu',4,'number of preprocessing threads per tower')

