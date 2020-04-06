#!/usr/bin/env bash

stage=2
#if [ $stage -le 0 ]; then
#  for loss in asoft soft ; do
#    python TrainAndTest/trans_dic2model.py \
#      --check-path Data/checkpoint/LoResNet10/spect/${loss} \
#      --loss-type ${loss} \
#      --epochs 20
#  done
#fi

#stage=10

if [ $stage -le 1 ]; then
  for loss in asoft soft  ; do
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss} \
      --resume Data/checkpoint/LoResNet10/spect/${loss}/checkpoint_1.pth \
      --loss-type ${loss}

  done
fi

if [ $stage -le 2 ]; then

  for loss in center amsoft ; do
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss} \
      --resume Data/checkpoint/LoResNet10/spect/soft/checkpoint_20.pth \
      --loss-type ${loss} \
      --lr 0.01 \
      --milestones 6 \
      --epochs 10

  done

fi