#!/usr/bin/env bash

stage=40

waited=0
while [ `ps 113458 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

#stage=1

if [ $stage -le 40 ]; then
  datasets=cnceleb
  model=DomResNet
  resnet_size=8
  for loss in soft; do
    python TrainAndTest/Spectrogram/train_domres_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/cnceleb/spect/dev_04 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/cnceleb/spect/test \
      --feat-format npy \
      --resnet-size 8 \
      --nj 10 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_04/${loss} \
      --resume Data/checkpoint/LoResNet8/${datasets}/spect_04/${loss}/checkpoint_1.pth \
      --channels 8,32,128 \
      --embedding-size-a 128 \
      --embedding-size-b 64 \
      --embedding-size-o 32 \
      --input-per-spks 192 \
      --num-valid 1 \
      --alpha 9 \
      --dom-ratio 0.5 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi