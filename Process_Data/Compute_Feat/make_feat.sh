#!/usr/bin/env bash

stage=3
# voxceleb1
if [ $stage -le 0 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64 \
      --out-set ${name}_kaldi \
      --feat-type fbank \
      --filter-type mel
  done
fi

if [ $stage -le 1 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64 \
      --out-set ${name}_linear \
      --feat-type fbank \
      --filter-type linear
  done
fi

#stage=4
if [ $stage -le 2 ]; then
  for name in reverb babble noise music ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_${name}_fb64/dev \
      --out-set dev_${name}

  done
fi

# vox1 spectrogram 257
if [ $stage -le 3 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect \
      --out-set ${name}_257 \
      --windowsize 0.025 \
      --nfft 512 \
      --feat-type spectrogram
  done
fi
stage=100

# sitw
if [ $stage -le 4 ]; then
  for name in dev eval ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --out-set ${name} \
      --feat-type spectrogram
  done
fi

# timit
if [ $stage -le 5 ]; then
  for name in train test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit \
      --out-set ${name}_spect \
      --feat-type spectrogram
  done
fi

if [ $stage -le 6 ]; then
  for name in train test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit \
      --out-set ${name}_fb64 \
      --feat-type fbank
  done
fi