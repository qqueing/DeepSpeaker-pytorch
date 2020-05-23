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

__author__ = 'Jonas Van Der Donckt'
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Define_Model.Pooling import AttentionStatisticPooling

"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""


class TimeDelayLayer_v1(nn.Module):
    def __init__(self, context, input_dim, output_dim, full_context=True):
        """
        Definition of context is the same as the way it's defined in the Peddinti paper. It's a list of integers, eg: [-2,2]
        By deault, full context is chosen, which means: [-2,2] will be expanded to [-2,-1,0,1,2] i.e. range(-2,3)
        """
        super(TimeDelayLayer_v1, self).__init__()
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
        x = x.squeeze(1)
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


# Implement of 'https://github.com/cvqluu/TDNN/blob/master/tdnn.py'
class TimeDelayLayer_v2(nn.Module):

    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1,
                 batch_norm=True, dropout_p=0.0, activation='relu'):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TimeDelayLayer_v2, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        if activation == 'relu':
            self.nonlinearity = nn.ReLU()
        elif activation == 'leakyrelu':
            self.nonlinearity = nn.LeakyReLU()

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def set_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = self.dropout_p

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({}) in ({})'.format(
            self.input_dim, d, str(x.shape))
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)

        x = self.nonlinearity(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        if self.dropout_p:
            x = self.drop(x)

        return x


class TDNN_v1(nn.Module):
    def __init__(self, context, input_dim, output_dim, node_num, full_context):
        super(TDNN_v1, self).__init__()
        self.tdnn1 = TimeDelayLayer_v1(context[0], input_dim, node_num[0], full_context[0])
        self.tdnn2 = TimeDelayLayer_v1(context[1], node_num[0], node_num[1], full_context[1])
        self.tdnn3 = TimeDelayLayer_v1(context[2], node_num[1], node_num[2], full_context[2])
        self.tdnn4 = TimeDelayLayer_v1(context[3], node_num[2], node_num[3], full_context[3])
        self.tdnn5 = TimeDelayLayer_v1(context[4], node_num[3], node_num[4], full_context[4])
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

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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


class TDNN_v2(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0.,
                 dropout_p=0.0, **kwargs):
        super(TDNN_v2, self).__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.alpha = alpha

        self.frame1 = TimeDelayLayer_v2(input_dim=self.input_dim, output_dim=512, context_size=5, dilation=1)
        self.frame2 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.frame3 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.frame4 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame5 = TimeDelayLayer_v2(input_dim=512, output_dim=1500, context_size=1, dilation=1)

        self.segment6 = nn.Sequential(
            nn.Linear(3000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.segment7 = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size)
        )
        self.classifier = nn.Linear(embedding_size, num_classes)
        # self.bn = nn.BatchNorm1d(num_classes)
        self.drop = nn.Dropout(p=self.dropout_p)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer_v2):
                # nn.init.normal(m.kernel.weight, mean=0., std=1.)
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity='relu')

    def l2_norm(self, input, alpha=1.0):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output * alpha

    def statistic_pooling(self, x):
        mean_x = x.mean(dim=1)
        std_x = x.var(dim=1, unbiased=False).add_(1e-12).sqrt()
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = dropout_p

    def forward(self, x):
        # pdb.set_trace()
        x = x.squeeze(1).float()
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        if self.dropout_p:
            x = self.drop(x)

        # print(x.shape)
        x = self.statistic_pooling(x)

        x = self.segment6(x)
        embedding_b = self.segment7(x)

        if self.alpha:
            embedding_b = self.l2_norm(embedding_b, self.alpha)

        logits = self.classifier(embedding_b)
        # logits = self.out_act(x)

        return logits, embedding_b

class ASTDNN(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim=24, dropout_p=0.0, **kwargs):
        super(ASTDNN, self).__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.input_dim = input_dim

        self.frame1 = TimeDelayLayer_v2(input_dim=self.input_dim, output_dim=512, context_size=5, dilation=1,
                                        dropout_p=dropout_p)
        self.frame2 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=2,
                                        dropout_p=dropout_p)
        self.frame3 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=3,
                                        dropout_p=dropout_p)
        self.frame4 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                        dropout_p=dropout_p)
        self.frame5 = TimeDelayLayer_v2(input_dim=512, output_dim=1500, context_size=1, dilation=1,
                                        dropout_p=dropout_p)

        self.attention_statistic = AttentionStatisticPooling(input_dim=1500, hidden_dim=64)

        self.segment6 = nn.Sequential(
            nn.Linear(3000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.segment7 = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size)
        )

        self.classifier = nn.Linear(embedding_size, num_classes)
        self.drop = nn.Dropout(p=self.dropout_p)

        # self.out_act = nn.Sigmoid()
        # self.relu = nn.LeakyReLU()
        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer_v2):
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity='relu')

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = self.dropout_p

        self.frame1.set_dropout(dropout_p)
        self.frame2.set_dropout(dropout_p)
        self.frame3.set_dropout(dropout_p)
        self.frame4.set_dropout(dropout_p)
        self.frame5.set_dropout(dropout_p)

    def forward(self, x):
        # pdb.set_trace()
        x = x.squeeze(1).float()
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        # print(x.shape)
        x = self.attention_statistic(x)
        embedding_a = self.segment6(x)

        if self.dropout_p:
            embedding_a = self.drop(embedding_a)

        embedding_b = self.segment7(embedding_a)

        if self.dropout_p:
            embedding_b = self.drop(embedding_b)

        logits = self.classifier(embedding_b)

        return logits, embedding_b

class ETDNN(nn.Module):
    def __init__(self, num_classes, embedding_size=256, batch_norm=True,
                 input_dim=80, dropout_p=0.0, **kwargs):
        super(ETDNN, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.dropout_p = dropout_p

        self.frame1 = TimeDelayLayer_v2(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine2 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame3 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=2,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine4 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame5 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=3,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine6 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame7 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=4,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame8 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame9 = TimeDelayLayer_v2(input_dim=512, output_dim=1500, context_size=1, dilation=1,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)

        # self.segment11 = nn.Linear(3000, embedding_size)
        # self.leakyrelu = nn.LeakyReLU()
        # self.batchnorm = nn.BatchNorm1d(embedding_size)
        self.segment11 = nn.Sequential(nn.Linear(3000, embedding_size),
                                       nn.LeakyReLU(),
                                       nn.BatchNorm1d(embedding_size))

        self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer_v2):
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity='leaky_relu')

    def statistic_pooling(self, x):
        mean_x = x.mean(dim=1)
        # std_x = x.std(dim=1)
        std_x = x.var(dim=1, unbiased=False).add_(1e-12).sqrt()
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p

        for m in self.modules():
            if isinstance(m, TimeDelayLayer_v2):
                m.set_dropout(dropout_p)

    def forward(self, x):
        # pdb.set_trace()
        if x.shape[1] == 1:
            x = x.squeeze(1).float()

        x = self.frame1(x)
        x = self.affine2(x)
        x = self.frame3(x)
        x = self.affine4(x)
        x = self.frame5(x)
        x = self.affine6(x)
        x = self.frame7(x)
        x = self.frame8(x)
        x = self.frame9(x)

        x = self.statistic_pooling(x)
        embeddings = self.segment11(x)

        logits = self.classifier(embeddings)

        return logits, embeddings
