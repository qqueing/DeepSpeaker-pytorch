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
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from Define_Model.model import ResSpeakerModel

__author__ = 'Jonas Van Der Donckt'


class TDNN(nn.Module):
    def __init__(self, context: list, input_channels: int, output_channels: int, full_context: bool = True):
        """
        Implementation of a 'Fast' TDNN layer by exploiting the dilation argument of the PyTorch Conv1d class
        Due to its fastness the context has gained two constraints:
            * The context must be symmetric
            * The context must have equal spacing between each consecutive element
        For example: the non-full and symmetric context {-3, -2, 0, +2, +3} is not valid since it doesn't have
        equal spacing; The non-full context {-6, -3, 0, 3, 6} is both symmetric and has an equal spacing, this is
        considered valid.
        :param context: The temporal context
        :param input_channels: The number of input channels
        :param output_channels: The number of channels produced by the temporal convolution
        :param full_context: Indicates whether a full context needs to be used
        """
        super(TDNN, self).__init__()
        self.full_context = full_context
        self.input_dim = input_channels
        self.output_dim = output_channels

        context = sorted(context)
        self.check_valid_context(context, full_context)

        if full_context:
            kernel_size = context[-1] - context[0] + 1 if len(context) > 1 else 1
            self.temporal_conv = weight_norm(nn.Conv1d(input_channels, output_channels, kernel_size))
        else:
            # use dilation
            delta = context[1] - context[0]
            self.temporal_conv = weight_norm(
                nn.Conv1d(input_channels, output_channels, kernel_size=len(context), dilation=delta))

    def forward(self, x):
        """
        :param x: is one batch of data, x.size(): [batch_size, input_channels, sequence_length]
            sequence length is the dimension of the arbitrary length data
        :return: [batch_size, output_dim, len(valid_steps)]
        """
        return self.temporal_conv(x)

    @staticmethod
    def check_valid_context(context: list, full_context: bool) -> None:
        """
        Check whether the context is symmetrical and whether and whether the passed
        context can be used for creating a convolution kernel with dil
        :param full_context: indicates whether the full context (dilation=1) will be used
        :param context: The context of the model, must be symmetric if no full context and have an equal spacing.
        """
        if full_context:
            assert len(context) <= 2, "If the full context is given one must only define the smallest and largest"
            if len(context) == 2:
                assert context[0] + context[-1] == 0, "The context must be symmetric"
        else:
            assert len(context) % 2 != 0, "The context size must be odd"
            assert context[len(context) // 2] == 0, "The context contain 0 in the center"
            if len(context) > 1:
                delta = [context[i] - context[i - 1] for i in range(1, len(context))]
                assert all(delta[0] == delta[i] for i in range(1, len(delta))), "Intra context spacing must be equal!"


# Create a TDNN layer
layer_context = [-2, 0, 2]
input = torch.ones(20, 257, 200)
input_n_feat = 257
con1 = torch.nn.Conv1d(257, 512, 3, stride=2)


from tensorboardX import SummaryWriter
tdnn_layer = TDNN(context=layer_context, input_channels=input_n_feat, output_channels=512, full_context=False)


# with SummaryWriter(comment='TDNN') as w:
#     model = tdnn_layer
#     w.add_graph(model, input, verbose=True)


# Run a forward pass; batch.size = [BATCH_SIZE, INPUT_CHANNELS, SEQUENCE_LENGTH]
outc = con1(input)
out = tdnn_layer(input)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

dummy_input = Variable(torch.rand(13, 1, 257, 32))


# model = ResSpeakerModel(resnet_size=34, embedding_size=512, num_classes=1211, feature_dim=64)

model = TDNN(context=layer_context, input_channels=input_n_feat, output_channels=512, full_context=False)
with SummaryWriter(comment='ResDeepSpeaker') as w:
    w.add_graph(model, (dummy_input, ))

print('')