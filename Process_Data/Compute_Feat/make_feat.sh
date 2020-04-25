#!/usr/bin/env bash

stage=9
# voxceleb1
if [ $stage -le 0 ]; then
  for name in dev test ; do
#    python Process_Data/Compute_Feat/make_feat_kaldi.py \
#      --nj 16 \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64 \
#      --out-set ${name}_noc \
#      --windowsize 0.025 \
#      --filters 64 \
#      --feat-type fbank

     python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb \
      --out-set ${name}_fb24 \
      --windowsize 0.02 \
      --nfft 320 \
      --feat-type fbank \
      --filter-type mel \
      --filters 24 \
      --feat-type fbank

    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb \
      --out-set ${name}_fb40 \
      --windowsize 0.02 \
      --nfft 320 \
      --feat-type fbank \
      --filter-type mel \
      --filters 40 \
      --feat-type fbank
  done
fi

if [ $stage -le 1 ]; then
  for name in dev test ; do
#    python Process_Data/Compute_Feat/make_feat_kaldi.py \
#      --nj 16 \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64 \
#      --out-set ${name}_noc \
#      --windowsize 0.025 \
#      --filters 64 \
#      --feat-type fbank

#     python Process_Data/Compute_Feat/make_feat_kaldi.py \
#      --nj 16 \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect \
#      --out-set ${name}_noc \
#      --windowsize 0.02 \
#      --nfft 320 \
#      --feat-type spectrogram

    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb \
      --out-set ${name}_dfb24 \
      --windowsize 0.02 \
      --nfft 320 \
      --feat-type fbank \
      --filter-type dnn.vox1 \
      --filters 24 \
      --feat-type fbank
  done
fi

#stage=100

if [ $stage -le 1 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb80 \
      --out-set ${name}_kaldi \
      --feat-type fbank \
      --filter-type mel \
      --filters 80 \
      --windowsize 0.025
  done
fi

if [ $stage -le 2 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pymfcc40 \
      --out-set ${name}_kaldi \
      --feat-type mfcc \
      --filters 40
  done
fi
#stage=100
if [ $stage -le 2 ]; then
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
if [ $stage -le 3 ]; then
  for name in reverb babble noise music ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_${name}_fb64/dev \
      --out-set dev_${name}

  done
fi

# vox1 spectrogram 257
if [ $stage -le 4 ]; then
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
#stage=100
#vox1 spectrogram 161
if [ $stage -le 5 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --nj 16 \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect \
      --out-set ${name}_noc \
      --windowsize 0.02 \
      --feat-type spectrogram
  done
fi


# sitw
if [ $stage -le 6 ]; then
  for name in dev eval ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --out-set ${name} \
      --feat-type spectrogram
  done
fi

# timit
if [ $stage -eq 7 ]; then
  for name in train test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect \
      --out-set ${name}_noc \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02
  done
fi

#stage=11
if [ $stage -le 8 ]; then
  for name in train test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
      --out-set ${name}_fb24 \
      --feat-type fbank \
      --filter-type mel \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 24
  done
fi

#stage=100
if [ $stage -le 9 ]; then
  for name in train test ; do
#    python Process_Data/Compute_Feat/make_feat_kaldi.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit \
#      --out-set ${name}_fb40_20 \
#      --filter-type mel \
#      --feat-type fbank \
#      --nfft 320 \
#      --windowsize 0.02 \
#      --filters 40

#    python Process_Data/Compute_Feat/make_feat_kaldi.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit \
#      --out-set ${name}_fb40_dnn_20 \
#      --filter-type dnn.timit \
#      --feat-type fbank \
#      --nfft 320 \
#      --windowsize 0.02 \
#      --filters 40
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
      --out-set ${name}_fb30 \
      --filter-type mel \
      --feat-type fbank \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 30

    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
      --out-set ${name}_dfb30_fix \
      --filter-type dnn.timit.fix \
      --feat-type fbank \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 30

    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb \
      --out-set ${name}_dfb30_var \
      --filter-type dnn.timit.var \
      --feat-type fbank \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 30

  done
fi

stage=100
#if [ $stage -le 10 ]; then
#  for name in train test ; do
#    python Process_Data/Compute_Feat/make_feat_kaldi.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit \
#      --out-set ${name}_mfcc_20 \
#      --filter-type mel \
#      --feat-type mfcc \
#      --nfft 320 \
#      --lowfreq 20 \
#      --windowsize 0.02 \
#      --filters 30 \
#      --numcep 24
#
#    python Process_Data/Compute_Feat/make_feat_kaldi.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit \
#      --out-set ${name}_mfcc_dnn_20 \
#      --filter-type dnn.timit \
#      --feat-type mfcc \
#      --nfft 320 \
#      --lowfreq 20 \
#      --windowsize 0.02 \
#      --filters 30 \
#      --numcep 24
#  done
#fi

stage=100
# libri
if [ $stage -le 11 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect \
      --out-set ${name}_noc \
      --feat-type spectrogram \
      --nfft 320 \
      --windowsize 0.02
  done
fi

#stage=10
if [ $stage -le 12 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb \
      --out-set ${name}_fb40 \
      --filter-type mel \
      --feat-type fbank \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 40

#     python Process_Data/Compute_Feat/make_feat_kaldi.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri \
#      --out-set ${name}_fb40_dnn_20 \
#      --filter-type dnn.timit \
#      --feat-type fbank \
#      --nfft 320 \
#      --windowsize 0.02 \
#      --filters 40
  done
fi

if [ $stage -le 13 ]; then
  for name in dev test ; do
    python Process_Data/Compute_Feat/make_feat_kaldi.py \
      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/${name} \
      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb \
      --out-set ${name}_fb24 \
      --filter-type mel \
      --feat-type fbank \
      --nfft 320 \
      --windowsize 0.02 \
      --filters 24

#     python Process_Data/Compute_Feat/make_feat_kaldi.py \
#      --data-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/${name} \
#      --out-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri \
#      --out-set ${name}_fb24_dnn_20 \
#      --filter-type dnn.timit \
#      --feat-type fbank \
#      --nfft 320 \
#      --windowsize 0.02 \
#      --filters 24
  done
fi

