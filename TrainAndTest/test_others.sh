#!/usr/bin/env bash

for loss in asoft soft ; do
  python TrainAndTest/test_sitw.py \
    --nj 12 \
    --check-path Data/checkpoint/LoResNet10/spect/${loss} \
    --veri-pairs 18000 \
    --loss-type ${loss} \
    --gpu-id 0 \
    --epochs 20
done

#python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
#    --check-path Data/checkpoint/LoResNet10/spect/amsoft \
#    --resume Data/checkpoint/LoResNet10/spect/soft/checkpoint_20.pth \
#    --loss-type amsoft \
#    --lr 0.01 \
#    --epochs 10