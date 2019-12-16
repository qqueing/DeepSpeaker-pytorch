 #!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: PLDA.py
@Time: 19-6-26 下午9:41
@Overview: 
Implement PLDA for speaker identification, Using PLDA in kaldi from command line.

"""
import numpy as np
import subprocess


# ivector-mean scp:Data/checkpoint/ResNet10/Fb_No/xvector.scp Data/checkpoint/ResNet10/Fb_No/mean.vec
# ivector_mean('Data/checkpoint/ResNet10/Fb_No/xvector.scp', 'Data/checkpoint/ResNet10/Fb_No/mean.vec')
def ivector_mean(xvect_scp, mean_vec):
    scp = 'scp:' + str(xvect_scp)
    mean_vec = str(mean_vec)
    ret = subprocess.Popen(['ivector-mean', scp, mean_vec], stdout=subprocess.PIPE)
    re_out = ret.stdout.readlines()

    if len(re_out)==0:
        print('Computing global ivector mean successes!')


# ivector-compute-lda --total-covariance-factor=0.0 --dim=200  "ark:ivector-subtract-global-mean scp:Data/checkpoint/ResNet10/Fb_No/xvector.scp ark:- |" ark:Data/checkpoint/ResNet10/Fb_No/utt2spk Data/checkpoint/ResNet10/Fb_No/transform.mat
# ivector_lda('Data/checkpoint/ResNet10/Fb_No/xvector.scp', 'Data/checkpoint/ResNet10/Fb_No/utt2spk', 'Data/checkpoint/ResNet10/Fb_No', dim=300)
def ivector_lda(xvect_scp, utt2spk, trans_mat_path, dim=200):
    xvect_scp = 'scp:' + str(xvect_scp)
    ark_utt2spk = 'ark:' + str(utt2spk)
    trans_mat = trans_mat_path + '/transform.mat'
    args_dim = '--dim=' + str(dim)

    ret = subprocess.Popen(['ivector-compute-lda', '--total-covariance-factor=0.0', args_dim,
                      "ark:ivector-subtract-global-mean {} ark:- |".format(xvect_scp),
                      ark_utt2spk,
                      trans_mat
                      ], stdout=subprocess.PIPE)
    re_out = ret.stdout.readlines()

    print(re_out)


ivector_lda('Data/checkpoint/ResNet10/Fb_No/xvector.scp', 'Data/checkpoint/ResNet10/Fb_No/utt2spk', 'Data/checkpoint/ResNet10/Fb_No', dim=300)



