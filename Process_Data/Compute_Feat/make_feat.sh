#!/usr/bin/env bash

for name in reverb babble noise music ; do
  python Process_Data/Compute_Feat/make_feat_kaldi.py \
    --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_${name}_fb64/dev \
    --out-set dev_$(name}

done