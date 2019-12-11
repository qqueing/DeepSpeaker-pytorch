#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ResNet.py
@Time: 2019/10/10 下午5:09
@Overview: Deep Speaker using Resnet with CNN, which is not ordinary Resnet.
This file define resnet in 'Deep Residual Learning for Image Recognition'

For all model, the pre_forward function is for extract vectors and forward for classification.
"""
import math
import pdb

from torch import nn
import torch
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import BasicBlock

from Define_Model.SoftmaxLoss import AngleLinear


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SimpleResNet(nn.Module):

    def __init__(self, layers, block=BasicBlock, num_classes=1000,
                 zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SimpleResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        num_filter = [16, 32, 64, 128]

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, num_filter[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(num_filter[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        num_filter = [16, 32, 64, 128]

        self.layer1 = self._make_layer(block, num_filter[0], layers[0])
        self.layer2 = self._make_layer(block, num_filter[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_filter[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_filter[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128 * block.expansion, num_filter[3])
        # self.norm = self.l2_norm(num_filter[3])
        self.alpha = 12

        self.fc2 = nn.Linear(num_filter[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal(m.weight, mean=0., std=1.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant(m.bn2.weight, 0)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def pre_forward(self, x):
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)

        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = torch.flatten(x, 1)
        # print(x.shape)
        return x

    def pre_forward_norm(self, x):
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = self.l2_norm(x)
        x = x * self.alpha

        return x

    def _forward(self, x):
        x = self.fc2(x)

        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[1, 1, 1, 1],
                 channels=[64, 128, 256, 512], num_classes=1000,
                 expansion=2, embedding=512, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.channels = channels
        self.inplanes = self.channels[0]
        self.conv1 = nn.Conv2d(1, self.channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.channels = channels

        self.layer1 = self._make_layer(block, self.channels[0], layers[0])
        self.layer2 = self._make_layer(block, self.channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.channels[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((expansion, 1))

        if self.channels[3] == 0:
            self.fc1 = nn.Linear(self.channels[2] * expansion, embedding)
        else:
            self.fc1 = nn.Linear(self.channels[3] * expansion, embedding)

        self.fc2 = nn.Linear(embedding, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def pre_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

    def forward(self, x):

        x = self.fc2(x)

        return x

class AResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[1, 1, 1, 1],
                 channels=[64, 128, 256, 512], num_classes=1000,
                 expansion=2, embedding=512, m=3, zero_init_residual=False):
        super(AResNet, self).__init__()
        self.expansion = expansion
        self.channels = channels
        self.inplanes = self.channels[0]
        self.conv1 = nn.Conv2d(1, self.channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.channels = channels

        self.layer1 = self._make_layer(block, self.channels[0], layers[0])
        self.layer2 = self._make_layer(block, self.channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.channels[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((expansion, 1))

        if self.channels[3] == 0:
            self.fc1 = nn.Linear(self.channels[2] * expansion, embedding)
        else:
            self.fc1 = nn.Linear(self.channels[3] * expansion, embedding)

        self.angle_linear = AngleLinear(in_features=embedding, out_features=num_classes, m=m)

        # self.fc2 = nn.Linear(embedding, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        logit = self.angle_linear(x)

        return logit, x
# model = SimpleResNet(block=BasicBlock, layers=[3, 4, 6, 3])
# input = torch.torch.randn(128,1,400,64)
# x_vectors = model.pre_forward(input)
# outputs = model(x_vectors)
# print('hello')
