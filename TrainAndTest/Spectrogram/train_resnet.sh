#!/usr/bin/env bash

stage=0
if [ $stage -le 0 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_resnet.py \
      --model ResNet \
      --resnet-size 18 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
      --embedding-size 512 \
      --batch-size 128 \
      --test-batch-size 32 \
      --nj 12 \
      --epochs 22 \
      --milestones 10,15 \
      --lr 0.1 \
      --veri-pairs 12800 \
      --check-path Data/checkpoint/ResNet/18/spect/${loss}_ \
      --resume Data/checkpoint/ResNet/18/spect/${loss}/checkpoint_1.pth \
      --loss-type ${loss}
  done
fi
