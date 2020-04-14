#!/usr/bin/env bash

if [ $stage -le 0 ]; then
  for model in LoResNet10 ; do
    python Lime/output_extract.py \
      --model ${model} \
      --epochs 19 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
      --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --check-path /home/yangwenhao/local/project/DeepSpeaker-pytorch/Data/checkpoint/LoResNet10/spect/soft \
      --extract-path Lime/${model} \
      --dropout-p 0.5 \
      --sample-utt 500

  done
fi

if [ $stage -le 1 ]; then
#  for model in LoResNet10 ; do
  python Lime/output_extract.py \
    --model LoResNet10 \
    --start-epochs 36 \
    --epochs 36 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --check-path /home/yangwenhao/local/project/DeepSpeaker-pytorch/Data/checkpoint/LoResNet10/spect_cmvn/center_dp25 \
    --extract-path Lime/LoResNet10 \
    --dropout-p 0 \
    --sample-utt 500

#  done
fi