#!/usr/bin/env bash

stage=0
if [ $stage -le 0 ]; then
  for loss in soft asoft amsoft center; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_resnet.py \
      --model ResNet \
      --resnet-size 18 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257 \
      --embedding-size 512 \
      --batch-size 128 \
      --test-batch-size 32 \
      --nj 12 \
      --epochs 24 \
      --milestones 10,15,20 \
      --lr 0.1 \
      --margin 0.35 \
      --s 30 \
      --m 4 \
      --veri-pairs 12800 \
      --check-path Data/checkpoint/ResNet/18/spect/${loss} \
      --resume Data/checkpoint/ResNet/18/spect/${loss}/checkpoint_1.pth \
      --loss-type ${loss}
  done
fi

TrainAndTest/Spectrogram/train_lores.sh