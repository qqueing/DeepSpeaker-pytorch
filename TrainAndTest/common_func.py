#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: common_func.py
@Time: 2019/12/16 6:36 PM
@Overview:
"""
import torch.optim as optim

def create_optimizer(parameters, optimizer, **kwargs):
    # setup optimizer
    if optimizer == 'sgd':
        opt = optim.SGD(parameters,
                              lr=kwargs['lr'],
                              momentum=kwargs['momentum'],
                              dampening=kwargs['dampening'],
                              weight_decay=kwargs['weight_decay'])

    elif optimizer == 'adam':
        opt = optim.Adam(parameters,
                               lr=kwargs['lr'],
                               weight_decay=kwargs['weight_decay'])

    elif optimizer == 'adagrad':
        opt = optim.Adagrad(parameters,
                                  lr=kwargs['lr'],
                                  lr_decay=kwargs['lr_decay'],
                                  weight_decay=kwargs['weight_decay'])

    return opt


def write_test_scalar(writer, epoch, **kwargs):
    writer.add_scalar('Test/Valid_Accuracy', kwargs['valid_accuracy'], epoch)
    writer.add_scalar('Test/EER', kwargs['eer'], epoch)
    writer.add_scalar('Test/Threshold', kwargs['eer_threshold'], epoch)