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
import inspect
import torch.optim as optim

def create_optimizer(parameters, optimizer, **kwargs):
    # setup optimizer
    parameters = filter(lambda p: p.requires_grad, parameters)
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


class AverageMeter(object):
    """Computes and stores the average and current value.
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def write_test_scalar(writer, epoch, **kwargs):
    writer.add_scalar('Test/Valid_Accuracy', kwargs['valid_accuracy'], epoch)
    writer.add_scalar('Test/EER', kwargs['eer'], epoch)
    writer.add_scalar('Test/Threshold', kwargs['eer_threshold'], epoch)

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]