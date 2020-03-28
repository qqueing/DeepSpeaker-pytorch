#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ConfigModel.py
@Time: 2020/3/28 5:11 PM
@Overview:
"""
import torch
import torch.nn as nn


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class ConfigModel(nn.Module):
    # model with config dict
    def __init__(self, **kwargs):
        super(ConfigModel).__init__()
        self.config = kwargs
        self.classifier = None

    def pre_forward(self, x):
        pass

    def forward(self, feat):
        x = self.classifier(feat)

        return x
