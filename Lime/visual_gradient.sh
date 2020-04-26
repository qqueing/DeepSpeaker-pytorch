#!/usr/bin/env bash

stage=1
if [ $stage -le 0 ]; then
  for model in LoResNet10 ; do
    python Lime/visual_gradient.py \
      --extract-path Data/gradient/LoResNet10/timit/spect_161/soft_var/LoResNet10/soft_dp0.00/epoch_15 \
      --feat-dim 161
  done
fi

#stage=10
if [ $stage -le 1 ]; then
  for model in LoResNet10 ; do
    python Lime/visual_gradient.py \
      --extract-path Data/gradient/LoResNet10/timit/spect/soft_fix/epoch_15 \
      --feat-dim 161

    python Lime/visual_gradient.py \
      --extract-path Data/gradient/LoResNet10/timit/spect/soft_var/epoch_15 \
      --feat-dim 161
  done
fi

stage=10
if [ $stage -le 2 ]; then
  for model in LoResNet10 ; do
    python Lime/visual_gradient.py \
      --extract-path Data/gradient/LoResNet10/timit/spect/soft_fix/LoResNet10/soft_dp0.00/epoch_15 \
      --feat-dim 161
  done
fi


if [ $stage -le 5 ]; then
#Data/gradient/LoResNet10/libri/spect/soft_128_0.25/epoch_15/
  for model in LoResNet10 ; do
    python Lime/visual_gradient.py \
      --extract-path Data/gradient/LoResNet10/libri/spect/soft_128_0.25/epoch_15 \
      --feat-dim 161
#    python Lime/visual_gradient.py \
#      --extract-path Data/gradient/LoResNet10/center_dp0.00/epoch_36 \
#      --feat-dim 161
  done
fi
stage=100
if [ $stage -le 10 ]; then
  for model in LoResNet10 ; do
    python Lime/visual_gradient.py \
      --extract-path Data/gradient/LoResNet10/libri/spect_161/soft_var_1500/LoResNet10/soft_dp0.00/epoch_15 \
      --feat-dim 161
#    python Lime/visual_gradient.py \
#      --extract-path Data/gradient/LoResNet10/center_dp0.00/epoch_36 \
#      --feat-dim 161
  done
fi