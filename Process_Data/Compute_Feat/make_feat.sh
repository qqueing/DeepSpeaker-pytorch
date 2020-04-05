#!/usr/bin/env bash

stage=3

# voxceleb1
if [ $stage -le 0 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-set ${name}

  done
fi

if [ $stage -le 1 ]; then
  for name in reverb babble noise music ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_${name}_fb64/dev \
      --out-set dev_${name}

  done
fi

# sitw
if [ $stage -le 3 ]; then
  for name in dev eval ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --out-set ${name} \
      --feat-type spectrogram
  done
fi