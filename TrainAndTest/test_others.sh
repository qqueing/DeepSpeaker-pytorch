#!/usr/bin/env bash

stage=5
if [ $stage -le 0 ]; then
  for loss in asoft soft ; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"

    python TrainAndTest/test_sitw.py \
      --nj 12 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss} \
      --veri-pairs 12800 \
      --loss-type ${loss} \
      --gpu-id 0 \
      --epochs 20
  done
fi

if [ $stage -le 5 ]; then
  model=LoResNet10
  for loss in soft ; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    python TrainAndTest/test_vox1.py \
      --nj 12 \
      --model ${model} \
      --resume Data/checkpoint/LoResNet10/spect_cmvn/soft_dp25/checkpoint_24.pth \
      --loss-type soft \
      --gpu-id 1
  done
fi

#python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
#    --check-path Data/checkpoint/LoResNet10/spect/amsoft \
#    --resume Data/checkpoint/LoResNet10/spect/soft/checkpoint_20.pth \
#    --loss-type amsoft \
#    --lr 0.01 \
#    --epochs 10

