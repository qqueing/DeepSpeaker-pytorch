#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ThinResNet.py
@Time: 2019/9/11 下午9:01
@Overview: Todo: projection mapping! 20190913
"""
import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, 2 * planes, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(2 * planes)

        self.downsample = downsample
        self.stride = stride

        self.linearprojection = nn.Parameter(torch.Tensor())

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ThinResNet(nn.Module):
    def __init__(self,
                 block=ResidualBlock,
                 layers=[1, 1, 1, 1],
                 embedding_size=None,
                 n_classes=1000
                 ):

        super(ThinResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=1, padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, 48, layers[0])
        self.layer2 = self._make_layer(block, 48, 96, layers[1])
        self.layer3 = self._make_layer(block, 96, 128, layers[2])
        self.layer4 = self._make_layer(block, 128, 256, layers[3])

        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(7, 1), stride=1, padding=(3, 0))

    def _make_layer(self, block, inplanes, planes, num_block, stride=1):

        layers = []
        layers.append(block(inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.maxpool(x)
        s = self.conv2(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x
