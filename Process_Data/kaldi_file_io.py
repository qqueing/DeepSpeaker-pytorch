#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: kaldi_file_io.py
@Time: 2019/12/10 下午9:28
@Overview:
"""
import kaldi_io
import numpy as np

def write_xvector_ark(uid, xvector, ark_file, scp_file):
    """
    :param uid:
    :param xvector: np.float32
    :param ark_file:
    :param scp_file:
    :return:
    """
    with open(scp_file, 'w') as scp:
        with open(ark_file, 'wb') as ark:
            for i in range(len(uid)):
                vec = xvector[i]
                len_vec = len(vec.tobytes())
                key = uid[i]
                kaldi_io.write_vec_flt(ark, vec, key=key)
                print(ark.tell())
                scp.write(str(uid[i]) + ' ' + str(ark_file) + ':' + str(ark.tell()-len_vec-10) + '\n')


# uid = ['A.J._Buckley-1zcIwhmdeo4-0001.wav', 'A.J._Buckley-1zcIwhmdeo4-0002.wav', 'A.J._Buckley-1zcIwhmdeo4-0003.wav', 'A.J._Buckley-7gWzIy6yIIk-0001.wav']
# xvector = np.random.randn(4, 512).astype(np.float32)
#
# ark_file = '../Data/xvector.ark'
# scp_file = '../Data/xvector.scp'
#
# write_xvector_ark(uid, xvector, ark_file, scp_file)

