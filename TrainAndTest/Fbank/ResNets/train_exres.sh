#!/usr/bin/env bash

stage=0
#stage=10
model=ExResNet

if [ $stage -le 0 ]; then
#  for loss in soft asoft ; do
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/dev_kaldi \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/test_kaldi \
      --feat-dim 64 \
      --nj 12 \
      --epochs 30 \
      --batch-size 80 \
      --lr 0.1 \
      --milestones 14,20,25 \
      --check-path Data/checkpoint/${model}/spect/${loss} \
      --resume Data/checkpoint/${model}/spect/${loss}/checkpoint_1.pth \
      --input-per-spks 240 \
      --num-valid 2 \
      --loss-type ${loss}
  done
fi
#stage=10

if [ $stage -le 1 ]; then
#  for loss in center amsoft ; do/
  for loss in center asoft; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/dev_kaldi \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/test_kaldi \
      --nj 12 \
      --feat-dim 64 \
      --batch-size 80 \
      --check-path Data/checkpoint/${model}/spect/${loss} \
      --resume Data/checkpoint/${model}/spect/soft/checkpoint_30.pth \
      --input-per-spks 240 \
      --loss-type ${loss} \
      --lr 0.01 \
      --loss-ratio 0.01 \
      --milestones 5,9 \
      --num-valid 2 \
      --epochs 12
  done

fi

#if [ $stage -le 2 ]; then
#
##  for loss in center amsoft ; do/
#  for kernel in '7,7' '3,7' '5,7' ; do
#    echo -e "\n\033[1;4;31m Training with kernel size ${kernel} \033[0m\n"
#    python TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
#      --nj 12 \
#      --epochs 18 \
#      --milestones 8,13,18 \
#      --check-path Data/checkpoint/LoResNet10/spect/kernel_${kernel} \
#      --kernel-size ${kernel}
#  done
#
#fi
