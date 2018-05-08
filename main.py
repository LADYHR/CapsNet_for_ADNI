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
from utils import get_batch_data

from config import cfg
from CapsNet import CapsNet

tf.logging.set_verbosity(tf.logging.INFO)


def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return fd_train_acc, fd_loss, fd_val_acc
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return fd_test_acc


def train(model, supervisor):
    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()  # 用于创建session的时候对session进行参数配置
    config.gpu_options.allow_growth = True  # 刚开始配置少量GPU内存，然后按需慢慢增加（不会释放内存，会导致碎片）
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print('Training for epoch ' + str(epoch) + '/' + str(cfg.epoch) + ':')
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(cfg.num_tr_batch), total=cfg.num_tr_batch, ncols=70, leave=False, unit='b'):
                global global_step
                global_step = epoch * cfg.num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:  # 每1个mini-batch进行一次total_loss、accuracy、train_summary的记录
                    _, loss, train_acc, summary_str = sess.run(
                        [model.train_op, model.margin_loss, model.accuracy, model.train_summary])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()  # 刷新缓冲区
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)  # 每个mini-batch进行一次模型的优化

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:  # 每2个mini-batch进行一次验证
                    val_acc = 0
                    for i in range(cfg.num_val_batch):
                        img, label = sess.run([model.val_img, model.val_label])
                        acc = sess.run(model.accuracy, {model.X: img,
                                                        model.labels: label})  # feed_dict用来临时替换掉一个op的输出结果
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * cfg.num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))
            # 如果没有saver，模型不会自动保存

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()

def evaluation(model, supervisor):
    fd_test_acc = save_to()
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))  # 重新导入模型
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(cfg.num_te_batch), total=cfg.num_te_batch, ncols=70, leave=False, unit='b'):
            img, label = sess.run([model.te_img, model.te_label])
            acc = sess.run(model.accuracy, {model.X: img, model.labels: label})
            test_acc += acc
        test_acc = test_acc / (cfg.batch_size * cfg.num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_accuracy.txt')


def main(_):
    tf.logging.info('Loading Graph...')
    model = CapsNet()
    tf.logging.info('Graph loaded!')
    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)
    if cfg.is_training:
        tf.logging.info('Start Training...')
        train(model, sv)
        tf.logging.info('Train done!')
    else:
        evaluation(model, sv)
        tf.logging.info('Test done!')
    print("Main programming finished!")


if __name__ == "__main__":
    tf.app.run()


