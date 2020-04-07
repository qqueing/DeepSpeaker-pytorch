#!/usr/bin/env bash

stage=0
if [ $stage -le 0 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_resnet20_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_257 \
      --embedding-size 128 \
      --batch-size 64 \
      --nj 12 \
      --epochs 30 \
      --milestones 14,24 \
      --veri-pairs 12800 \
      --check-path Data/checkpoint/ResNet20/spect_257/${loss} \
      --resume Data/checkpoint/ResNet20/spect_257/${loss}/checkpoint_1.pth \
      --loss-type ${loss}
  done
fi