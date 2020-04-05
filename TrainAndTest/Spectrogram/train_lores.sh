#!/usr/bin/env bash

for loss in asoft soft center ; do
  python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
    --nj 12 \
    --check-path Data/checkpoint/LoResNet10/spect/${loss} \
    --resume Data/checkpoint/LoResNet10/spect/${loss}/checkpoint_1.pth \
    --loss-type ${loss}

done

python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
    --check-path Data/checkpoint/LoResNet10/spect/amsoft \
    --resume Data/checkpoint/LoResNet10/spect/soft/checkpoint_20.pth \
    --loss-type amsoft \
    --lr 0.001 \
    --epochs 10