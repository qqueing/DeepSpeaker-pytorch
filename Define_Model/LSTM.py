#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: LSTM.py
@Time: 2020/5/13 11:28 AM
@Overview:
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTM_End(nn.Module):
    def __init__(self, input_dim, num_class, batch_size,
                 embedding_size=256,
                 hidden_shape=128, num_lstm=2, dropout_p=0.2):
        super(LSTM_End, self).__init__()

        self.num_lstm = num_lstm
        self.hidden_shape = hidden_shape
        self.lstm_layer = nn.LSTM(input_size=input_dim,
                                  hidden_size=hidden_shape,
                                  num_layers=self.num_lstm,
                                  batch_first=True,
                                  dropout=dropout_p)

        self.h0 = nn.Parameter(torch.rand(self.num_lstm, batch_size, hidden_shape), requires_grad=False)
        self.c0 = nn.Parameter(torch.rand(self.num_lstm, batch_size, hidden_shape), requires_grad=False)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)

        self.fc1 = nn.Sequential(nn.Linear(hidden_shape, embedding_size),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(embedding_size))

        self.fc2 = nn.Linear(embedding_size, num_class)

    def varlen_forward(self, input, length):

        out, (_, _) = self.lstm_layer(input, (self.h0, self.c0))
        out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)

        out_pad_shape = out_pad.shape
        out_pad_idx = torch.ones(out_pad_shape[0], 1, out_pad_shape[2])

        if out_pad.is_cuda:
            out_len = (out_len - 1)
            out_pad_idx = out_pad_idx.cpu()
            out_pad = out_pad.cpu()

        for n in range(len(out_pad)):
            out_pad_idx[n][0] = out_pad_idx[n][0] * out_len[n]

        # pdb.set_trace()
        rnn_out = out_pad.gather(dim=1, index=out_pad_idx.long()).squeeze()

        # rnn_last =
        spk_vec = self.fc1(rnn_out.cuda())
        logits = self.fc2(spk_vec)

        return spk_vec, logits

    def forward(self, input):
        """
        :param input: should be features with fixed length
        :return:
        """

        out, (_, _) = self.lstm_layer(input, (self.h0, self.c0))
        rnn_out = out[:, -1, :].squeeze()
        # rnn_last =
        feats = self.fc1(rnn_out.cuda())
        logits = self.fc2(feats)

        return logits, feats


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, num_class, batch_size,
                 embedding_size=256,
                 hidden_shape=128, num_lstm=2,
                 dropout_p=0.2, attention_dim=64):
        super(AttentionLSTM, self).__init__()

        self.num_lstm = num_lstm
        self.hidden_shape = hidden_shape
        self.lstm_layer = nn.LSTM(input_size=input_dim,
                                  hidden_size=hidden_shape,
                                  num_layers=self.num_lstm,
                                  batch_first=True,
                                  dropout=dropout_p)

        self.h0 = nn.Parameter(torch.rand(self.num_lstm, batch_size, hidden_shape), requires_grad=False)
        self.c0 = nn.Parameter(torch.rand(self.num_lstm, batch_size, hidden_shape), requires_grad=False)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)

        self.attention_linear = nn.Linear(hidden_shape, attention_dim)
        self.attention_activation = nn.Sigmoid()
        self.attention_vector = nn.Parameter(torch.rand(attention_dim, 1))
        self.attention_soft = nn.Tanh()

        self.fc1 = nn.Sequential(nn.Linear(hidden_shape, embedding_size),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(embedding_size))

        self.fc2 = nn.Linear(embedding_size, num_class)

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

    def forward(self, input):
        """
        :param input: should be features with fixed length
        :return:
        """

        out, (_, _) = self.lstm_layer(input, (self.h0, self.c0))
        # pdb.set_trace()
        rnn_out = self.attention_layer(out)

        feats = self.fc1(rnn_out)
        logits = self.fc2(feats)

        return logits, feats
