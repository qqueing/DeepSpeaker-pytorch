#!/usr/bin/env bash


model=ETDNN

for loss in soft ; do
  python TrainAndTest/Fbank/TDNNs/train_etdnn_kaldi.py \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb80/dev_kaldi \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb80/test_kaldi \
    --check-path Data/checkpoint/${model}/fbank80/soft \
    --resume Data/checkpoint/${model}/fbank80/soft/checkpoint_1.pth
    --epochs 20 \
    --milestones 10,15  \
    --feat-dim 80 \
    --embedding-size 256 \
    --num-valid 2 \
    --loss-type soft \
    --lr 0.01

done