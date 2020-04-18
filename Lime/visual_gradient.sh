#!/usr/bin/env bash

stage=0
if [ $stage -le 0 ]; then
  for model in LoResNet10 ; do
    python Lime/visual_gradient.py \
      --extract-path Data/xvector/LoResNet10/timit/spect_161/soft/LoResNet10/soft_dp0.00/epoch_15 \
      --feat-dim 161
  done
fi
