#!/usr/bin/env bash

stage=0
#stage=10

if [ $stage -le 0 ]; then
  for loss in soft asoft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 18 \
      --milestones 8,13,18 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss} \
      --resume Data/checkpoint/LoResNet10/spect/${loss}/checkpoint_1.pth \
      --loss-type ${loss}
  done
fi

if [ $stage -le 1 ]; then
#  for loss in center amsoft ; do/
  for loss in amsoft center ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss} \
      --resume Data/checkpoint/LoResNet10/spect/soft/checkpoint_20.pth \
      --loss-type ${loss} \
      --lr 0.01 \
      --loss-ratio 0.1 \
      --milestones 4 \
      --epochs 8
  done

fi

if [ $stage -le 2 ]; then

#  for loss in center amsoft ; do/
  for kernel in '7,7' '3,7' '5,7' ; do
    echo -e "\n\033[1;4;31m Training with kernel size ${kernel} \033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 18 \
      --milestones 8,13,18 \
      --check-path Data/checkpoint/LoResNet10/spect/kernel_${kernel} \
      --resume Data/checkpoint/LoResNet10/spect/kernel_${kernel}/checkpoint_20.pth \
      --epochs 20 \
      --kernel-size ${kernel}
  done

fi
