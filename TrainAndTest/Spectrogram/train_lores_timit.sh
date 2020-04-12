#!/usr/bin/env bash

#stage=3
stage=5

if [ $stage -le 0 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect \
      --nj 14 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/${loss}_re \
      --resume Data/checkpoint/LoResNet10/timit_spect/${loss}_re/checkpoint_1.pth \
      --channels 4,16,64 \
      --statis-pooling \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 2 \
      --weight-decay 0.001 \
      --dropout-p 0.5 \
      --loss-type ${loss}
  done
fi
#stage=10
if [ $stage -le 1 ]; then
  for loss in asoft amsoft center ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect \
      --nj 12 \
      --epochs 8 \
      --lr 0.1 \
      --loss-ratio 0.01 \
      --milestones 3,6 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/${loss} \
      --resume Data/checkpoint/LoResNet10/timit_spect/soft/checkpoint_1.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 2 \
      --dropout-p 0.5 \
      --weight-decay 0.001 \
      --loss-type ${loss} \
      --m 4 \
      --lambda-max 0.2 \
      --margin 0.3 \
      --s 30
  done
fi


#stage=2
if [ $stage -le 2 ]; then
#  for loss in center amsoft amsoft ; do/
  for loss in amsoft ; do
    echo -e "\n\033[1;4;31m Finetuning with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect \
      --nj 12 \
      --epochs 6 \
      --milestones 4 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/${loss}_sch5 \
      --resume Data/checkpoint/LoResNet10/timit_spect/soft/checkpoint_12.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --lr 0.1 \
      --loss-ratio 0.01 \
      --num-valid 2 \
      --weight-decay 0.001 \
      --loss-type ${loss} \
      --m 4 \
      --lambda-max 0.2 \
      --margin 0.3 \
      --s 30
  done
fi

if [ $stage -le 3 ]; then
#  for loss in center amsoft ; do/
  for kernel in '3,3' '3,5' ; do
    echo -e "\n\033[1;4;31m Training with kernel size ${kernel} \033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect \
      --nj 12 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --resume None \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 2 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/kernel_${kernel} \
      --weight-decay 0.001 \
      --kernel-size ${kernel}
  done
fi

#stage=5
if [ $stage -le 4 ]; then
#  for loss in center amsoft ; do/
#  for p in 0.1 0.2 0.5 ; do
  for p in 0.1 0.2 0.5 ; do
    echo -e "\n\033[1;4;31m Training with dropout-${p} \033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect \
      --nj 12 \
      --epochs 6 \
      --milestones 4 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/dropout_${p} \
      --resume Data/checkpoint/LoResNet10/timit_spect/soft/checkpoint_10.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 2 \
      --lr 0.01 \
      --weight-decay 0.001 \
      --dropout-p ${p}
  done

fi

if [ $stage -le 5 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_var.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect_noc \
      --nj 14 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/${loss}_var \
      --resume Data/checkpoint/LoResNet10/timit_spect/${loss}_var/checkpoint_1.pth \
      --channels 4,16,64 \
      --statis-pooling \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 2 \
      --weight-decay 0.001 \
      --dropout-p 0.5 \
      --loss-type ${loss}
  done
fi

if [ $stage -le 5 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect_noc \
      --nj 14 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet10/timit_spect/${loss}_fix \
      --resume Data/checkpoint/LoResNet10/timit_spect/${loss}_fix/checkpoint_1.pth \
      --channels 4,16,64 \
      --statis-pooling \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 2 \
      --weight-decay 0.001 \
      --dropout-p 0.5 \
      --loss-type ${loss}
  done
fi
exit 0;
