#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: CNN.py
@Time: 2020/5/18 12:16 PM
@Overview:
"""
from torch import nn


class AlexNet(nn.Module):

    def __init__(self, num_classes, embedding_size=128, time_dim=2, avg_size=2,
                 dropout_p=0.0, **kwargs):
        super(AlexNet, self).__init__()
        self.max = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout_p = dropout_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((time_dim, avg_size))

        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 * time_dim * avg_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max(x)

        x = self.conv2(x)
        x = self.max(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.conv6(x)
        x = self.max(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embedding_a = self.fc1(x)
        embedding_b = self.fc2(embedding_a)

        logits = self.classifier(embedding_b)

        return logits, embedding_b
