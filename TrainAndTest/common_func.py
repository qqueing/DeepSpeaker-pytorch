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

from Define_Model.ResNet import LocalResNet, ResNet20, ExporingResNet, ResNet, SimpleResNet
from Define_Model.TDNN import ASTDNN, XVectorTDNN


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


# ALSTM  ASiResNet34  ExResNet34  LoResNet10  ResNet20  SiResNet34  SuResCNN10  TDNN

__factory = {
    'LoResNet10': LocalResNet,
    'ResNet20': ResNet20,
    'SiResNet34': SimpleResNet,
    'ExResNet34': ExporingResNet,
    'ResNet': ResNet,
    'ASTDNN': ASTDNN,
    'TDNN': XVectorTDNN,
    'ETDNN': ETDNN,
}


def create_model(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))

    return __factory[name](**kwargs)


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

