#!/usr/bin/env bash

stage=6
#stage=10
#waited=0
#while [ `ps 105999 | wc -l` -eq 2 ]; do
#  sleep 60
#  waited=$(expr $waited + 1)
#  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
#done

if [ $stage -le 0 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3x3\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 20 \
      --milestones 10,15 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_dp33 \
      --resume Data/checkpoint/LoResNet10/spect/${loss}_dp33/checkpoint_20.pth \
      --kernel-size 3,3 \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.5

    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5x5\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 20 \
      --milestones 10,15 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_dp \
      --resume Data/checkpoint/LoResNet10/spect/${loss}_dp/checkpoint_1.pth \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.5
  done
fi

stage=6
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
stage=6
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

if [ $stage -le 5 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3x3\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 10 \
      --milestones 4,7 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_dp33_0.1 \
      --resume Data/checkpoint/LoResNet10/spect/${loss}_dp33/checkpoint_20.pth \
      --kernel-size 3,3 \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.1

    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5x5\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 10 \
      --milestones 4,7 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_dp01 \
      --resume Data/checkpoint/LoResNet10/spect/${loss}/checkpoint_1.pth \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.1
  done
fi

if [ $stage -le 6 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3x3\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 20 \
      --milestones 10,15 \
      --resnet-size 10 \
      --check-path Data/checkpoint/LoResNet10/spectrogram/${loss} \
      --resume Data/checkpoint/LoResNet10/spectrogram/${loss}/checkpoint_20.pth \
      --channels 32,128,256,512 \
      --kernel-size 3,3 \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.5
  done
fi
