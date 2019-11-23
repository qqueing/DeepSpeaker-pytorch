#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: SAP.py
@Time: 2019/8/7 下午4:47
@Overview: Self-attentive encoding layer implement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
HIDDEN_DIM = 3
FEATURE_WIDTH = 2
FEATURE_LENGTH = 3

class SelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.hidden_dim = HIDDEN_DIM
        self.linear_operate = nn.Linear(in_dim, out_dim)
        self.tanh_operate = nn.Tanh()
        self.exp_operate = nn.Softmax(dim=1)
        self.context_vector = nn.Parameter(torch.randn(out_dim, out_dim), requires_grad=True)

        # self.projection = nn.Sequential(
        #     nn.Linear(in_dim, 300),
        #     nn.Tanh(),
        #     nn.Linear(300, 300)
        # )
        # self.center_vector = nn.Parameter(torch.randn(FEATURE_WIDTH, FEATURE_LENGTH), requires_grad=True)

    def forward(self, input):
        # batch_size = encoder_outputs.size(0)
        # (B, L, H) -> (B , L, 1)
        # energy = self.projection(encoder_outputs)
        # weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        # outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        mlp_out = self.tanh_operate(self.linear_operate(input))
        mul_con = torch.matmul(self.context_vector, mlp_out)

        weight_para = self.exp_operate(mul_con.data)

        outputs = torch.sum(torch.matmul(weight_para, input), dim=1)

        return outputs, weight_para


class SelfAttentive(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=2)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=2)
        return outputs, weights

sa = SelfAttentive(hidden_dim=257)
a = torch.ones((1, 1, 345, 257))
out, wei = sa(a)
print(out.shape, wei.shape)



