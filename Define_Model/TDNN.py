#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: TDNN.py
@Time: 2019/8/28 上午10:54
@Overview: Implement TDNN

fork from:
https://github.com/jonasvdd/TDNN/blob/master/tdnn.py
"""
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F

__author__ = 'Jonas Van Der Donckt'


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import math

"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""

class TDNN(nn.Module):
    def __init__(self, context, input_dim, output_dim, full_context=True):
        """
        Definition of context is the same as the way it's defined in the Peddinti paper. It's a list of integers, eg: [-2,2]
        By deault, full context is chosen, which means: [-2,2] will be expanded to [-2,-1,0,1,2] i.e. range(-2,3)
        """
        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.check_valid_context(context)
        self.kernel_width, context = self.get_kernel_width(context, full_context)
        self.register_buffer('context',torch.LongTensor(context))
        self.full_context = full_context
        stdv = 1./math.sqrt(input_dim)
        self.kernel = nn.Parameter(torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(0,stdv))
        self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0,stdv))
        # self.cuda_flag = False

    def forward(self, x):
        """
        x is one batch of data
        x.size(): [batch_size, sequence_length, input_dim]
        sequence length is the length of the input spectral data (number of frames) or if already passed through the convolutional network, it's the number of learned features
        output size: [batch_size, output_dim, len(valid_steps)]
        """
        # Check if parameters are cuda type and change context
        # if type(self.bias.data) == torch.cuda.FloatTensor and self.cuda_flag == False:
        #     self.context = self.context.cuda()
        #     self.cuda_flag = True
        conv_out = self.special_convolution(x, self.kernel, self.context, self.bias)
        return conv_out

    def special_convolution(self, x, kernel, context, bias):
        """
        This function performs the weight multiplication given an arbitrary context. Cannot directly use convolution because in case of only particular frames of context, one needs to select only those frames and perform a convolution across all batch items and all output dimensions of the kernel.
        """
        # pdb.set_trace()
        x = x.squeeze()
        input_size = x.size()
        assert len(input_size) == 3, 'Input tensor dimensionality is incorrect. Should be a 3D tensor'
        [batch_size, input_dim, input_sequence_length] = input_size
        #x = x.transpose(1,2).contiguous() # [batch_size, input_dim, input_length]

        # Allocate memory for output
        valid_steps = self.get_valid_steps(self.context, input_sequence_length)
        #xs = torch.Tensor(self.bias.data.new(batch_size, kernel.size()[0], len(valid_steps)))
        xs = torch.zeros((batch_size, kernel.size()[0], len(valid_steps)))

        if torch.cuda.is_available():
            xs = Variable(xs.cuda())
        # Perform the convolution with relevant input frames
        # pdb.set_trace()
        for c, i in enumerate(valid_steps):
            features = torch.index_select(x, 2, Variable(context+i))
            # torch.index_selec:
            # Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
            # The returned tensor has the same number of dimensions as the original tensor (input). The dim th dimension has the same
            # size as the length of index; other dimensions have the same size as in the original tensor.
            xs[:,:,c] = F.conv1d(features, kernel, bias=bias)[:,:,0]

        return xs

    @staticmethod
    def check_valid_context(context): #检查context是否合理
        # here context is still a list
        assert context[0] <= context[-1], 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0],context[-1]+1) #确定一个context的范围
        return len(context), context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        """
        Return the valid index frames considering the context.
        确定给定长度的序列，卷积之后的长度，及其帧
        :param context:
        :param input_sequence_length:
        :return:
        """

        start = 0 if context[0] >= 0 else -1*context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        return range(start, end)

class Time_Delay(nn.Module):
    def __init__(self, context, input_dim, output_dim, node_num, full_context):
        super(Time_Delay, self).__init__()
        self.tdnn1 = TDNN(context[0], input_dim, node_num[0], full_context[0])
        self.tdnn2 = TDNN(context[1], node_num[0], node_num[1], full_context[1])
        self.tdnn3 = TDNN(context[2], node_num[1], node_num[2], full_context[2])
        self.tdnn4 = TDNN(context[3], node_num[2], node_num[3], full_context[3])
        self.tdnn5 = TDNN(context[4], node_num[3], node_num[4], full_context[4])
        self.fc1 = nn.Linear(node_num[5], node_num[6])
        self.fc2 = nn.Linear(node_num[6], node_num[7])
        self.fc3 = nn.Linear(node_num[7], output_dim)
        self.batch_norm1 = nn.BatchNorm1d(node_num[0])
        self.batch_norm2 = nn.BatchNorm1d(node_num[1])
        self.batch_norm3 = nn.BatchNorm1d(node_num[2])
        self.batch_norm4 = nn.BatchNorm1d(node_num[3])
        self.batch_norm5 = nn.BatchNorm1d(node_num[4])
        self.batch_norm6 = nn.BatchNorm1d(node_num[6])
        self.batch_norm7 = nn.BatchNorm1d(node_num[7])
        self.input_dim = input_dim
        self.output_dim = output_dim

    def statistic_pooling(self, x):
        mean_x = x.mean(dim=2)
        std_x = x.std(dim=2)
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std

    def pre_forward(self, x):
        a1 = F.relu(self.batch_norm1(self.tdnn1(x)))
        a2 = F.relu(self.batch_norm2(self.tdnn2(a1)))
        a3 = F.relu(self.batch_norm3(self.tdnn3(a2)))
        a4 = F.relu(self.batch_norm4(self.tdnn4(a3)))
        a5 = F.relu(self.batch_norm5(self.tdnn5(a4)))

        a6 = self.statistic_pooling(a5)
        x_vectors = F.relu(self.batch_norm6(self.fc1(a6)))

        return x_vectors

    def forward(self, x):
        # a7 = self.pre_forward(x)
        a8 = F.relu(self.batch_norm7(self.fc2(x)))
        output = self.fc3(a8)
        return output


# Create a TDNN layer
# layer_context = [-2, 0, 2]
# input = torch.ones(20, 257, 200)
# input_n_feat = 257
# con1 = torch.nn.Conv1d(257, 512, 3, stride=2)


# from tensorboardX import SummaryWriter
# tdnn_layer = TDNN(context=layer_context, input_channels=input_n_feat, output_channels=512, full_context=False)


# with SummaryWriter(comment='TDNN') as w:
#     model = tdnn_layer
#     w.add_graph(model, input, verbose=True)


# Run a forward pass; batch.size = [BATCH_SIZE, INPUT_CHANNELS, SEQUENCE_LENGTH]
# outc = con1(input)
# out = tdnn_layer(input)
#
# class Net2(nn.Module):
#     def __init__(self):
#         super(Net2, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         x = F.log_softmax(x, dim=1)
#         return x
#
# dummy_input = Variable(torch.rand(13, 1, 257, 32))


# model = ResSpeakerModel(resnet_size=34, embedding_size=512, num_classes=1211, feature_dim=64)
#
# model = TDNN(context=layer_context, input_channels=input_n_feat, output_channels=512, full_context=False)
# with SummaryWriter(comment='ResDeepSpeaker') as w:
#     w.add_graph(model, (dummy_input, ))
#
# print('')