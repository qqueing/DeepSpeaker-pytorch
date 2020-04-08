#!/usr/bin/env bash

stage=1
#stage=10

if [ $stage -le 0 ]; then
  for loss in soft asoft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect \
      --nj 8 \
      --epochs 10 \
      --milestones 4,8 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/${loss} \
      --resume Data/checkpoint/LoResNet10/timit_spect/${loss}/checkpoint_1.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 128 \
      --num-valid 2 \
      --weight-decay 0.001 \
      --loss-type ${loss}
  done
fi

if [ $stage -le 1 ]; then
#  for loss in center amsoft ; do/
  for loss in amsoft asoft ; do
    echo -e "\n\033[1;4;31m Finetuning with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect \
      --nj 8 \
      --epochs 6 \
      --milestones 3 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/${loss}_fine \
      --resume Data/checkpoint/LoResNet10/timit_spect/soft/checkpoint_10.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 128 \
      --lr 0.01 \
      --loss-ratio 0.01 \
      --num-valid 2 \
      --weight-decay 0.001 \
      --loss-type ${loss} \
      --margin 0.35 \
      --s 30
  done

fi

stage=3

if [ $stage -le 2 ]; then
#  for loss in center amsoft ; do/
  for kernel in '7,3' ; do
    echo -e "\n\033[1;4;31m Training with kernel size ${kernel} \033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect \
      --nj 8 \
      --epochs 10 \
      --milestones 4,8 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/${loss} \
      --resume None \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 128 \
      --num-valid 2 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/kernel_${kernel} \
      --weight-decay 0.001 \
      --kernel-size ${kernel}
  done

fi

if [ $stage -le 3 ]; then
#  for loss in center amsoft ; do/
  for p in 0.1 0.2 0.25 ; do
    echo -e "\n\033[1;4;31m Training with dropout-${p} \033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect \
      --nj 8 \
      --epochs 6 \
      --milestones 3 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/dropout_${p} \
      --resume Data/checkpoint/LoResNet10/timit_spect/soft/checkpoint_10.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 128 \
      --num-valid 2 \
      --weight-decay 0.001 \
      --dropout-p ${p}
  done

fi
