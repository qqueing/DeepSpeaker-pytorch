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

import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

from Define_Model.SoftmaxLoss import AngleLinear


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1,
                     stride=stride, bias=False)

class SimpleResNet(nn.Module):

    def __init__(self, layers, block=BasicBlock,
                 num_classes=1000,
                 embedding_size=128,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SimpleResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.embedding_size=embedding_size
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
        self.conv1 = nn.Conv2d(1, num_filter[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(num_filter[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        num_filter = [16, 32, 64, 128]

        self.layer1 = self._make_layer(block, num_filter[0], layers[0])
        self.layer2 = self._make_layer(block, num_filter[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_filter[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_filter[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128 * block.expansion, embedding_size)
        # self.norm = self.l2_norm(num_filter[3])
        self.alpha = 12

        self.fc2 = nn.Linear(embedding_size, num_classes)

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

        # pdb.set_trace()
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


# Analysis of Length Normalization in End-to-End Speaker Verification System
# https://arxiv.org/abs/1806.03209
class ExporingResNet(nn.Module):

    def __init__(self, layers, block=BasicBlock, input_frames=300, num_classes=1000, embedding_size=128,
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, **kwargs):
        super(ExporingResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.embedding_size = embedding_size
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
        self.conv1 = nn.Conv2d(1, num_filter[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(num_filter[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        num_filter = [16, 32, 64, 128]

        self.layer1 = self._make_layer(block, num_filter[0], layers[0])
        self.layer2 = self._make_layer(block, num_filter[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_filter[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_filter[3], layers[3], stride=2)

        # [64, 128, 8, 37]
        time_dim = int(np.ceil(input_frames / 8))
        self.avgpool = nn.AdaptiveAvgPool2d((1, time_dim))
        # 300 is the length of features

        self.fc1 = nn.Sequential(
            nn.Linear(num_filter[3] * time_dim, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        self.alpha = 12
        self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # \nn.init.normal(m.weight, mean=0., std=1.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the residual branch
        # starts with zeros, and each residual block behaves like an identity.
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

    def _forward(self, x):
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
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = self.l2_norm(x)
        feat = x * self.alpha

        x = self.classifier(x)

        return x, feat

    # Allow for accessing forward method in a inherited class
    forward = _forward

class AttenSiResNet(nn.Module):

    def __init__(self, layers, block=BasicBlock,
                 num_classes=1000,
                 embedding_size=128,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(AttenSiResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.embedding_size = embedding_size
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
        self.fc1 = nn.Linear(128 * block.expansion, embedding_size)
        # self.norm = self.l2_norm(num_filter[3])
        self.alpha = 12

        self.fc2 = nn.Linear(embedding_size, num_classes)

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

        self.attention_dim = num_filter[3]
        self.attention_linear = nn.Linear(num_filter[3], self.attention_dim)
        self.attention_activation = nn.Sigmoid()
        self.attention_vector = nn.Parameter(torch.rand(self.attention_dim, 1))
        self.attention_soft = nn.Tanh()

    def attention_layer(self, x):
        """
        :param x:   [length,feat_dim] vector
        :return:   [feat_dim] vector
        """
        fx = self.attention_activation(self.attention_linear(x))
        vf = fx.matmul(self.attention_vector)
        alpha = self.attention_soft(vf)

        alpha_ht = x.mul(alpha)
        w = torch.sum(alpha_ht, dim=-2)

        return w

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

    def att_forward_norm(self, x):
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

        x = x.view(x.size(0), x.size(1), -1)
        x = torch.transpose(x, 1, 2)

        x = self.attention_layer(x)

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

    def __init__(self, resnet_size=18, embedding_size=512, block=BasicBlock,
                 channels=[64, 128, 256, 512], num_classes=1000,
                 expansion=4, zero_init_residual=False, **kwargs):
        super(ResNet, self).__init__()

        resnet_layer = {10: [1, 1, 1, 1],
                        18: [2, 2, 2, 2],
                        34: [3, 4, 6, 3],
                        50: [3, 4, 6, 3],
                        101: [3, 4, 23, 3]}

        layers = resnet_layer[resnet_size]
        self.layers = layers

        self.expansion = expansion
        self.channels = channels
        self.inplanes = self.channels[0]
        self.conv1 = nn.Conv2d(1, self.channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.channels = channels

        self.layer1 = self._make_layer(block, self.channels[0], layers[0])
        self.layer2 = self._make_layer(block, self.channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.channels[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((expansion, 1))

        if self.layers[3] == 0:
            self.fc1 = nn.Sequential(
                nn.Linear(self.channels[2] * expansion, embedding_size),
                nn.BatchNorm1d(embedding_size)
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(self.channels[3] * expansion, embedding_size),
                nn.BatchNorm1d(embedding_size)
            )

        self.fc2 = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the residual
        # branch starts with zeros, and each residual block behaves like an identity.
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

        if self.layers[3] != 0:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        feat = self.fc1(x)
        logits = self.fc2(feat)

        return logits, feat

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

# M. Hajibabaei and D. Dai, “Unified hypersphere embedding for speaker recognition,”
# arXiv preprint arXiv:1807.08312, 2018.
class Block3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Block3x3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet20(nn.Module):
    def __init__(self, num_classes=1000, embedding_size=128, dropout_p=0.0,
                 block=BasicBlock, input_frames=300, **kwargs):
        super(ResNet20, self).__init__()
        self.dropout_p = dropout_p
        self.inplanes = 1
        self.layer1 = self._make_layer(Block3x3, planes=64, blocks=1, stride=2)

        self.inplanes = 64
        self.layer2 = self._make_layer(Block3x3, planes=128, blocks=1, stride=2)

        self.inplanes = 128
        self.layer3 = self._make_layer(BasicBlock, 128, 1)

        self.inplanes = 128
        self.layer4 = self._make_layer(Block3x3, planes=256, blocks=1, stride=2)

        self.inplanes = 256
        self.layer5 = self._make_layer(BasicBlock, 256, 3)

        self.inplanes = 256
        self.layer6 = self._make_layer(Block3x3, planes=512, blocks=1, stride=2)

        self.inplanes = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Sequential(
            nn.Linear(17 * self.inplanes, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        self.classifier = nn.Linear(embedding_size, num_classes)

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.dropout_p != 0:
            x = self.dropout(x)

        feat = self.fc1(x)

        logits = self.classifier(feat)

        return logits, feat


class LocalResNet(nn.Module):
    """
    Define the ResNet model with A-softmax and AM-softmax loss.
    Added dropout as https://github.com/nagadomi/kaggle-cifar10-torch7 after average pooling and fc layer.
    """

    def __init__(self, embedding_size, num_classes, block=BasicBlock,
                 resnet_size=8,
                 channels=[64, 128, 256], dropout_p=0.,
                 kernal_size=5, padding=2, **kwargs):
        super(LocalResNet, self).__init__()
        resnet_type = {8: [1, 1, 1, 0],
                       10: [1, 1, 1, 1],
                       18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3]}

        layers = resnet_type[resnet_size]
        self.layers = layers
        self.dropout_p = dropout_p

        self.embedding_size = embedding_size
        self.relu = nn.LeakyReLU()
        # self.relu = ReLU(inplace=True)

        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=1)
        self.layer1 = self._make_layer(block, channels[0], layers[0])

        self.inplanes = channels[1]
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernal_size, stride=2, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.layer2 = self._make_layer(block, channels[1], layers[1])

        self.inplanes = channels[2]
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernal_size, stride=2, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.layer3 = self._make_layer(block, channels[2], layers[2])

        if layers[3] != 0:
            assert len(channels) == 4
            self.inplanes = channels[3]
            self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernal_size, stride=2, padding=padding,
                                   bias=False)
            self.bn4 = nn.BatchNorm2d(channels[3])
            self.layer4 = self._make_layer(block=block, planes=channels[3], blocks=layers[3])

        # self.avg_pool = nn.AdaptiveAvgPool2d([4, 1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 4))
        self.dropout = nn.Dropout(self.dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(self.inplanes * 4, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        self.classifier = nn.Linear(self.embedding_size, num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # self.weight = nn.Parameter(torch.Tensor(embedding_size, num_classes))  # 本层权重
        # self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # 初始化权重，在第一维度上做normalize

    def l2_norm(self, input, alpha=1.0):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output * alpha

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
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.layer4(x)

        if self.dropout_p > 0:
            x = self.dropout(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.l2_norm(x, alpha=12)

        logits = self.classifier(x)

        return logits, x
