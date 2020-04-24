#!/usr/bin/env bash

#stage=3
stage=6

waited=0
while [ `ps 128196 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 0 ]; then
  datasets=timit
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_wcmvn \
      --nj 14 \
      --epochs 14 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet10/${datasets}/spect_wcmvn/${loss} \
      --resume Data/checkpoint/LoResNet10/${datasets}/spect_wcmvn/${loss}/checkpoint_1.pth \
      --channels 4,16,64 \
      --statis-pooling \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --loss-type ${loss}

    python TrainAndTest/Spectrogram/train_lores10_var.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_wcmvn \
      --nj 14 \
      --epochs 14 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet10/${datasets}/spect_wcmvn/${loss}_var \
      --resume Data/checkpoint/LoResNet10/${datasets}/spect_wcmvn/${loss}_var/checkpoint_1.pth \
      --channels 4,16,64 \
      --statis-pooling \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --loss-type ${loss}
  done
fi

stage=6
if [ $stage -le 6 ]; then
  datasets=libri
  model=LoResNet10
#  --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
#  --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_wcmvn \
      --nj 14 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet10/${datasets}/spect_wcmvn/${loss}_128_01 \
      --resume Data/checkpoint/LoResNet10/${datasets}/spect_wcmvn/${loss}_128_01/checkpoint_1.pth \
      --channels 4,32,128 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --alpha 12 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.1 \
      --loss-type ${loss}

#    python TrainAndTest/Spectrogram/train_lores10_var.py \
#      --model ${model} \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
#      --nj 14 \
#      --epochs 15 \
#      --lr 0.1 \
#      --milestones 7,11 \
#      --check-path Data/checkpoint/LoResNet10/${datasets}/spect/${loss}_var \
#      --resume Data/checkpoint/LoResNet10/${datasets}/spect/${loss}_var/checkpoint_1.pth \
#      --channels 4,16,64 \
#      --statis-pooling \
#      --embedding-size 128 \
#      --input-per-spks 256 \
#      --num-valid 2 \
#      --weight-decay 0.001 \
#      --dropout-p 0.25 \
#      --loss-type ${loss}
  done
fi

exit 0;
