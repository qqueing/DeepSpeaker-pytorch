#!/usr/bin/env bash

stage=60
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
#    python Lime/visual_gradient.py \
#      --extract-path Data/gradient/LoResNet10/timit/spect/soft_fix/epoch_15 \
#      --feat-dim 161

    python Lime/visual_gradient.py \
      --extract-path Data/gradient/LoResNet10/timit/spect/soft_var/epoch_15 \
      --feat-dim 161
  done
fi

#stage=200
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
#    python Lime/visual_gradient.py \
#      --extract-path Data/gradient/LoResNet10/libri/spect/soft/epoch_15 \
#      --feat-dim 161

    python Lime/visual_gradient.py \
      --extract-path Data/gradient/LoResNet10/libri/spect_noc/soft/epoch_15 \
      --feat-dim 161
#    python Lime/visual_gradient.py \
#      --extract-path Data/gradient/LoResNet10/center_dp0.00/epoch_36 \
#      --feat-dim 161
  done
fi
#stage=100
if [ $stage -le 10 ]; then
  python Lime/visual_gradient.py \
      --extract-path Data/gradient/LoResNet10/vox1/spect/soft_wcmvn/epoch_24 \
      --feat-dim 161

#  for loss in soft amsoft center ; do
#    python Lime/visual_gradient.py \
#      --extract-path Data/gradient/LoResNet10/spect/${loss}_wcmvn/epoch_38 \
#      --feat-dim 161
#  done
fi
#stage=100

if [ $stage -le 20 ]; then
    python Lime/visual_gradient.py \
      --extract-path Data/gradient/ResNet20/vox1/spect_256_wcmvn/soft_wcmvn/epoch_24 \
      --feat-dim 257
fi

#stage=100
if [ $stage -le 30 ]; then
#    python Lime/visual_gradient.py \
#      --extract-path Data/gradient/ExResNet34/vox1/fb64_wcmvn/soft_var/epoch_30 \
#      --feat-dim 64 \
#      --acoustic-feature fbank

    python Lime/visual_gradient.py \
      --extract-path Data/gradient/SiResNet34/vox1/fb64_wcmvn/soft_fix/epoch_40 \
      --feat-dim 64 \
      --acoustic-feature fbank

fi
#stage=100
if [ $stage -le 40 ]; then
    python Lime/visual_gradient.py \
      --extract-path Data/gradient/SiResNet34/vox1/fb64_kaldi/soft_fix/epoch_26 \
      --feat-dim 64 \
      --acoustic-feature fbank

#    python Lime/visual_gradient.py \
#      --extract-path Data/gradient/TDNN/fb40_wcmvn/soft/epoch_18 \
#      --feat-dim 40 \
#      --acoustic-feature fbank

fi
if [ $stage -le 50 ]; then
    python Lime/visual_gradient.py \
      --extract-path Data/gradient/SiResNet34/vox1/fb64_cmvn/soft/epoch_21 \
      --feat-dim 64 \
      --acoustic-feature fbank

#    python Lime/visual_gradient.py \
#      --extract-path Data/gradient/TDNN/fb40_wcmvn/soft/epoch_18 \
#      --feat-dim 40 \
#      --acoustic-feature fbank

fi

if [ $stage -le 60 ]; then
    python Lime/visual_gradient.py \
      --extract-path Data/gradient/LoResNet18/cnceleb/spect/soft_dp25/epoch_24 \
      --feat-dim 161 \
      --acoustic-feature spectrogram

#    python Lime/visual_gradient.py \
#      --extract-path Data/gradient/TDNN/fb40_wcmvn/soft/epoch_18 \
#      --feat-dim 40 \
#      --acoustic-feature fbank

fi
