#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: SoftmaxLoss.py
@Time: 2019/8/5 下午5:29
@Overview:
"AngleLinear" and "AngleSoftmaxLoss" Fork from
https://github.com/woshildh/a-softmax_pytorch/blob/master/a_softmax.py.

"AMSoftmax" Fork from
https://github.com/CoinCheung/pytorch-loss/blob/master/amsoftmax.py

"AngularSoftmax" is completed based on the two loss.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

__all__=["AngleLinear", "AngleSoftmaxLoss", "AngularSoftmax", "AMSoftmax"]

class AngleLinear(nn.Module):
    def __init__(self, in_planes, out_planes, m=4):
        super(AngleLinear,self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.m = m

        # The input should be [batch_size, in_plane]. The weight is trying to transform the dimension to [batch_size, out_plane].
        self.weight = nn.Parameter(torch.Tensor(in_planes, out_planes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.cos_function=[
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x,
        ]

    def forward(self, x):
        '''
        inputs:
            x: [batch, in_planes]
        return:
            cos_x: [batch, out_planes]
            phi_x: [batch, out_planes]
        '''
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5) #[batch, out_planes]
        x_modulus = x.pow(2).sum(1).pow(0.5) #[batch]
        w_modulus = w.pow(2).sum(0).pow(0.5) #[out_planes]

        # get w@x=||w||*||x||*cos(theta)
        inner_wx = x.mm(w) # [batch,out_planes]
        cos_theta = (inner_wx/x_modulus.view(-1,1))/w_modulus.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        # get cos(m*theta)
        cos_m_theta = self.cos_function[self.m](cos_theta)
        theta = Variable(cos_theta.data.acos())

        #get k, theta is in [ k*pi/m , (k+1)*pi/m ]
        k = (self.m * theta / math.pi).floor()
        minus_one = k*0 - 1

        # get phi_theta = -1^k*cos(m*theta)-2*k
        phi_theta = (minus_one ** k) * cos_m_theta - 2 * k

        # get cos_x and phi_x
        # cos_x = cos(theta)*||x||
        # phi_x = phi(theta)*||x||
        cos_x = cos_theta * x_modulus.view(-1,1)
        phi_x = phi_theta * x_modulus.view(-1,1)
        return cos_x , phi_x

class AngleSoftmaxLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleSoftmaxLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, inputs, target):
        '''
        inputs:
            cos_x: [batch, classes_num]
            phi_x: [batch, classes_num]
            target: LongTensor,[batch]
        return:
            loss:scalar
        '''
        self.it += 1
        cos_x,phi_x = inputs
        target = target.view(-1, 1)

        # get one_hot mat
        index = cos_x.data * 0.0 #size=(B,Classnum)
        index.scatter_(1, target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        # set lamb, change the rate of softmax and A-softmax
        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it))

        # get a-softmax and softmax mat
        output = cos_x * 1
        output[index] -= (cos_x[index] * 1.0/(+self.lamb))
        output[index] += (phi_x[index] * 1.0/(self.lamb))

        # get loss, which is equal to Cross Entropy.
        logpt = F.log_softmax(output, dim=1) #[batch,classes_num]
        logpt = logpt.gather(1, target) #[batch]
        pt = logpt.data.exp()
        loss = -1 * logpt * (1-pt)**self.gamma
        loss = loss.mean()

        return loss


class AngularSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 num_classes,
                 m=4,
                 gamma=0):
        super(AngularSoftmax, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.m = m
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.W = torch.nn.Parameter(torch.randn(in_feats, num_classes), requires_grad=True).cuda()
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal(self.W, gain=1)

        self.cos_function = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        ]

    def forward(self, x, label):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # pdb.set_trace()
        w = self.W.renorm(2, 1, 1e-5).mul(1e5) #[batch, out_planes]
        x_modulus = x.pow(2).sum(1).pow(0.5) #[batch]
        w_modulus = w.pow(2).sum(0).pow(0.5) #[out_planes]

        # get w@x=||w||*||x||*cos(theta)
        # w = w.cuda()
        inner_wx = x.mm(w) # [batch,out_planes]
        cos_theta = (inner_wx/x_modulus.view(-1,1))/w_modulus.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        # get cos(m*theta)
        cos_m_theta = self.cos_function[self.m](cos_theta)
        theta = Variable(cos_theta.data.acos())

        #get k, theta is in [ k*pi/m , (k+1)*pi/m ]
        k = (self.m * theta / math.pi).floor()
        minus_one = k*0 - 1

        # get phi_theta = -1^k*cos(m*theta)-2*k
        phi_theta = (minus_one ** k) * cos_m_theta - 2 * k

        # get cos_x and phi_x
        # cos_x = cos(theta)*||x||
        # phi_x = phi(theta)*||x||
        cos_x = cos_theta * x_modulus.view(-1,1)
        phi_x = phi_theta * x_modulus.view(-1,1)

        target = label.view(-1, 1)

        # get one_hot mat
        index = cos_x.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        # set lamb, change the rate of softmax and A-softmax
        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))

        # get a-softmax and softmax mat
        output = cos_x * 1
        # output[index] -= (cos_x[index] * 1.0 / (+self.lamb))
        # output[index] += (phi_x[index] * 1.0 / (self.lamb))
        output[index] -= cos_x[index]
        output[index] += phi_x[index]
        loss = self.ce(output, label)

        return loss


class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.3,
                 s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True).cuda()
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal(self.W, gain=1)

    def forward(self, x, label):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # pdb.set_trace()
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        lb_view = label.view(-1, 1)

        if lb_view.is_cuda:
            lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view.data, self.m)

        if x.is_cuda:
            delt_costh = Variable(delt_costh.cuda())

        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m

        loss = self.ce(costh_m_s, label)

        return loss


# Testing those Loss Classes
# a = Variable(torch.Tensor([[1., 1., 3.],
#                   [1., 2., 0.],
#                   [1., 4., 3.],
#                   [5., 0., 3.]]).cuda())
#
# a_label = Variable(torch.LongTensor([2, 1, 1, 0]).cuda())
#
# linear = AngleLinear(in_planes=3, out_planes=3, m=4)
# Asoft = AngleSoftmaxLoss()
# a_linear = linear(a)
# a_asoft = Asoft(a_linear, a_label)
# print("axsoftmax loss is {}".format(a_asoft))
#
# asoft = AngularSoftmax(in_feats=3, num_classes=3)
# a_loss = asoft(a, a_label)
# print("amsoftmax loss is {}".format(a_loss))
#
# amsoft = AMSoftmax(in_feats=3)
# am_a = amsoft(a, a_label)
# print("amsoftmax loss is {}".format(am_a))