#!/usr/bin/env bash


for model in LoResNet10 ; do
  python Lime/output_extract.py \
    --model ${model} \
    --epochs 19 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --check-path /home/yangwenhao/local/project/DeepSpeaker-pytorch/Data/checkpoint/LoResNet10/spect/soft \
    --extract-path Lime/${model} \
    --sample-utt 500

done