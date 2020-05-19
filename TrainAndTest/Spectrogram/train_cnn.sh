#!/usr/bin/env bash

stage=0

if [ $stage -le 1 ]; then
  model=AlexNet
  for loss in soft ; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Training ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
      --input-per-spks 224 \
      --nj 12 \
      --epochs 18 \
      --embedding-size 128 \
      --avg-size 2 \
      --milestones 10,14 \
      --check-path Data/checkpoint/${model}/spect/${loss} \
      --resume Data/checkpoint/${model}/spect/${loss}/checkpoint_29.pth \
      --loss-type ${loss} \
      --lr 0.01 \
      --num-valid 2 \
      --gpu-id 1 \
      --dropout-p 0.0
  done
fi