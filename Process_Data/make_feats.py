#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: make_feats.py
@Time: 2019/9/24 下午8:32
@Overview: make feats for missing speakers dataset using multi threads.
"""
from __future__ import print_function
import os
import pathlib
import pdb
import threading

from Process_Data.audio_processing import make_Fbank, conver_to_wav

missing_spks = ['id06996', 'id07001', 'id07002', 'id07008', 'id07017', 'id07031', 'id07032', 'id07043', 'id07044', 'id07057', 'id07062', 'id07063', 'id07065', 'id07072', 'id07080', 'id07084', 'id07085', 'id07117', 'id07118', 'id07133', 'id07135', 'id07148', 'id07153', 'id07165', 'id07179', 'id07181', 'id07182', 'id07183', 'id07186', 'id07187', 'id07200', 'id07205', 'id07206', 'id07220', 'id07232', 'id07243', 'id07256', 'id07273', 'id07275', 'id07278', 'id07293', 'id07299', 'id07305', 'id07308', 'id07334', 'id07341', 'id07342', 'id07352', 'id07356', 'id07367', 'id07368', 'id07375', 'id07398', 'id07415', 'id07416', 'id07418', 'id07423', 'id07429', 'id07430', 'id07434', 'id07437', 'id07442', 'id07459', 'id07460', 'id07484', 'id07500', 'id07508', 'id07512', 'id07524', 'id07527', 'id07528', 'id07537', 'id07540', 'id07546', 'id07548', 'id07550', 'id07554', 'id07558', 'id07561', 'id07562', 'id07578', 'id07587', 'id07588', 'id07626', 'id07628', 'id07636', 'id07640', 'id07646', 'id07669', 'id07673', 'id07681', 'id07683', 'id07694', 'id07697', 'id07703', 'id07711', 'id07714', 'id07730', 'id07751', 'id07752', 'id07755', 'id07756', 'id07759', 'id07774', 'id07783', 'id07786', 'id07800', 'id07814', 'id07816', 'id07826', 'id07828', 'id07834', 'id07838', 'id07840', 'id07845', 'id07853', 'id07856', 'id07882', 'id07884', 'id07885', 'id07898', 'id07901', 'id07914', 'id07926', 'id07941', 'id07954', 'id07967', 'id07968', 'id07972', 'id07973', 'id07974', 'id07977', 'id07981', 'id07990', 'id07996', 'id08037', 'id08043', 'id08046', 'id08048', 'id08049', 'id08066', 'id08074', 'id08078', 'id08115', 'id08119', 'id08123', 'id08126', 'id08132', 'id08138', 'id08139', 'id08144', 'id08145', 'id08147', 'id08159', 'id08166', 'id08171', 'id08173', 'id08190', 'id08204', 'id08213', 'id08241', 'id08244', 'id08246', 'id08258', 'id08260', 'id08261', 'id08264', 'id08271', 'id08277', 'id08283', 'id08300', 'id08306', 'id08314', 'id08320', 'id08321', 'id08337', 'id08349', 'id08354', 'id08372', 'id08394', 'id08395', 'id08397', 'id08404', 'id08421', 'id08424', 'id08426', 'id08437', 'id08445', 'id08453', 'id08477', 'id08478', 'id08499', 'id08516', 'id08520', 'id08530', 'id08534', 'id08536', 'id08547', 'id08573', 'id08580', 'id08588', 'id08589', 'id08598', 'id08602', 'id08603', 'id08606', 'id08607', 'id08616', 'id08617', 'id08622', 'id08629', 'id08634', 'id08645', 'id08648', 'id08665', 'id08668', 'id08669', 'id08674', 'id08684', 'id08687', 'id08695', 'id08698', 'id08702', 'id08704', 'id08721', 'id08732', 'id08741', 'id08742', 'id08762', 'id08773', 'id08774', 'id08775', 'id08781', 'id08785', 'id08789', 'id08805', 'id08821', 'id08840', 'id08862', 'id08866', 'id08868', 'id08892', 'id08898', 'id08901', 'id08909', 'id08916', 'id08919', 'id08920', 'id08928', 'id08930', 'id08938', 'id08971', 'id08974', 'id08981', 'id09008', 'id09012', 'id09016', 'id09023', 'id09027', 'id09029', 'id09036', 'id09038', 'id09062', 'id09065', 'id09071', 'id09075', 'id09091', 'id09095', 'id09138', 'id09152', 'id09171', 'id09175', 'id09179', 'id09186', 'id09192', 'id09199', 'id09210', 'id09212', 'id09217', 'id09218', 'id09225', 'id09232', 'id09234', 'id09238']
# pdb.set_trace()
num_make = 0

def make_feats_spks(spk_id):

    data_root = pathlib.Path('/home/cca01/work2019/Data/voxceleb2/dev/aac/{}'.format(spk_id))
    data_dir = pathlib.Path('/home/cca01/work2019/Data/voxceleb2/dev/aac')

    # print('\nspeaker is %s' % str(spk_id))

    # all the paths of wav files
    # dev/acc/spk_id/utt_group/wav_id.wav
    all_abs_path = list(data_root.glob('*/*.wav'))
    all_rel_path = [str(pathlib.Path.relative_to(path, data_dir)).rstrip('.wav') for path in all_abs_path]

    num_pro = 0
    for datum in all_rel_path:
        pdb.set_trace()
        # Data/Voxceleb1/
        # /data/voxceleb/voxceleb1_wav/
        # pdb.set_trace()
        filename = str(data_dir) + '/' +datum + '.wav'
        write_path = 'Data/Voxceleb2/dev/aac/' + datum + '.npy'

        if os.path.exists(filename):
            make_Fbank(filename=filename, write_path=write_path)
        # convert the audio format for m4a.
        # elif os.path.exists(filename.replace('.wav', '.m4a')):
            # conver_to_wav(filename.replace('.wav', '.m4a'),
            #               write_path=args.dataroot + '/voxceleb2/' + datum['filename'] + '.wav')
            #
            # make_Fbank(filename=filename,
            #            write_path=write_path)

        # print('\rThread: {} \t processed {:2f}% {}/{}.'.format(threadid, num_pro / len(all_rel_path), num_pro, len(all_rel_path)), end='\r')
        num_pro += 1



class MyThread(threading.Thread):
    def __init__(self, spk_ids, threadid):
        super(MyThread, self).__init__()  # 重构run函数必须要写
        self.spk_ids = spk_ids
        self.threadid = threadid

    def run(self):
        global num_make
        for spk_id in self.spk_ids:
            make_feats_spks(spk_id)
            num_make += 1
            print('\t{:4d} of speakers making feats completed!'.format(num_make))

if __name__ == "__main__":
    num_spk = len(missing_spks)
    trunk = int(num_spk / 4)

    for i in range(0, 4):
        j = (i+1)*trunk
        if i==3:
            j=num_spk

        print(i*trunk, j)
        # MyThread(missing_spks[i:j], i).start()






