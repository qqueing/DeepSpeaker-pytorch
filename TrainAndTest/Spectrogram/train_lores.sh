#!/usr/bin/env bash

stage=2
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
  for loss in center ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss} \
      --resume Data/checkpoint/LoResNet10/spect/soft/checkpoint_18.pth \
      --loss-type ${loss} \
      --lr 0.01 \
      --loss-ratio 0.01 \
      --milestones 4 \
      --epochs 10
  done
fi


if [ $stage -le 2 ]; then
#  for loss in center amsoft ; do/
  for loss in amsoft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_s30 \
      --resume Data/checkpoint/LoResNet10/spect/soft/checkpoint_18.pth \
      --loss-type ${loss} \
      --margin 0.35 \
      --s 30 \
      --loss-ratio 0.01 \
      --lr 0.01 \
      --milestones 4 \
      --epochs 8
  done
fi

if [ $stage -le 3 ]; then
#  for loss in center amsoft ; do/
  for loss in asoft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_m4 \
      --resume Data/checkpoint/LoResNet10/spect/soft/checkpoint_18.pth \
      --loss-type ${loss} \
      --m 4 \
      --lambda-max 1000 \
      --milestones 4 \
      --epochs 8
  done
fi
stage=10
# kernel size trianing
if [ $stage -le 4 ]; then
  for kernel in '3,3' '3,7' '5,7' ; do
    echo -e "\n\033[1;4;31m Training with kernel size ${kernel} \033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 18 \
      --milestones 8,13,18 \
      --resume Data/checkpoint/LoResNet10/spect/kernel_${kernel}/checkpoint_18.pth \
      --check-path Data/checkpoint/LoResNet10/spect/kernel_${kernel} \
      --kernel-size ${kernel}
  done

fi
