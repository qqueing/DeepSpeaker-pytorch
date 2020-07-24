#!/usr/bin/env bash

stage=30

waited=0
while [ `ps 113458 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

#stage=1
if [ $stage -le 0 ]; then
  for loss in soft ; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5x5\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
      --nj 12 \
      --epochs 24 \
      --resnet-size 8 \
      --embedding-size 128 \
      --milestones 10,15,20 \
      --channels 64,128,256 \
      --check-path Data/checkpoint/LoResNet8/spect/${loss}_wcmvn \
      --resume Data/checkpoint/LoResNet8/spect/${loss}_wcmvn/checkpoint_20.pth \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.25
  done
fi

#stage=100


if [ $stage -le 1 ]; then
#  for loss in center amsoft ; do/
  for loss in asoft amsoft center; do
    echo -e "\n\033[1;4;31m Finetuning with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
      --nj 12 \
      --resnet-size 8 \
      --epochs 14 \
      --milestones 6,10 \
      --embedding-size 128 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_wcmvn \
      --resume Data/checkpoint/LoResNet10/spect/soft_wcmvn/checkpoint_24.pth \
      --loss-type ${loss} \
      --loss-ratio 0.001 \
      --lr 0.01 \
      --lambda-max 1200 \
      --margin 0.35 \
      --s 15 \
      --m 3 \
      --num-valid 2 \
      --dropout-p 0.25
  done
fi

#stage=100
# kernel size trianing
if [ $stage -le 4 ]; then
  for kernel in '3,3' '3,7' '5,7' ; do
    echo -e "\n\033[1;4;31m Training with kernel size ${kernel} \033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 18 \
      --milestones 8,13,18 \
      --resume Data/checkpoint/LoResNet10/spect/kernel_${kernel}/checkpoint_18.pth \
      --check-path Data/checkpoint/LoResNet10/spect/kernel_${kernel} \
      --kernel-size ${kernel}
  done

fi

if [ $stage -le 5 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3x3\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 10 \
      --milestones 4,7 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_dp33_0.1 \
      --resume Data/checkpoint/LoResNet10/spect/${loss}_dp33/checkpoint_20.pth \
      --kernel-size 3,3 \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.1

    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5x5\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 10 \
      --milestones 4,7 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_dp01 \
      --resume Data/checkpoint/LoResNet10/spect/${loss}/checkpoint_1.pth \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.1
  done
fi

if [ $stage -le 6 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Continue Training with ${loss} kernel 3x3\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 4 \
      --resnet-size 10 \
      --check-path Data/checkpoint/LoResNet10/spectrogram/${loss} \
      --resume Data/checkpoint/LoResNet10/spectrogram/${loss}/checkpoint_20.pth \
      --channels 32,128,256,512 \
      --kernel-size 3,3 \
      --lr 0.0001 \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.5
  done
fi

if [ $stage -le 7 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3x3\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 24 \
      --milestones 10,15,20 \
      --resnet-size 10 \
      --check-path Data/checkpoint/LoResNet10/spectrogram/${loss}_64 \
      --resume Data/checkpoint/LoResNet10/spectrogram/${loss}_64/checkpoint_20.pth \
      --channels 64,128,256,512 \
      --kernel-size 3,3 \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.25
  done
fi


if [ $stage -le 15 ]; then
  for loss in soft ; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3,3\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
      --input-per-spks 224 \
      --nj 12 \
      --epochs 24 \
      --resnet-size 18 \
      --embedding-size 128 \
      --kernel-size 3,3 \
      --avg-size 4 \
      --milestones 10,15,20 \
      --channels 64,128,256,256 \
      --check-path Data/checkpoint/LoResNet18/spect/${loss}_dp25 \
      --resume Data/checkpoint/LoResNet18/spect/${loss}_dp25/checkpoint_1.pth \
      --loss-type ${loss} \
      --lr 0.1 \
      --num-valid 2 \
      --gpu-id 1 \
      --dropout-p 0.25
  done

#  for loss in amsoft center asoft ; do # 32,128,512; 8,32,128
#    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5x5\033[0m\n"
#    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
#      --model LoResNet10 \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
#      --input-per-spks 224 \
#      --nj 12 \
#      --epochs 12 \
#      --resnet-size 10 \
#      --embedding-size 128 \
#      --kernel-size 3,3 \
#      --milestones 6,10 \
#      --channels 64,128,256,512 \
#      --check-path Data/checkpoint/LoResNet10/spect/${loss}_dp25 \
#      --resume Data/checkpoint/LoResNet10/spect/soft_dp25/checkpoint_24.pth \
#      --loss-type ${loss} \
#      --finetune \
#      --lr 0.01 \
#      --num-valid 2 \
#      --margin 0.35 \
#      --s 40 \
#      --m 4 \
#      --loss-ratio 0.01 \
#      --gpu-id 0 \
#      --dropout-p 0.25
#  done
fi

if [ $stage -le 20 ]; then
  dataset=aishell2

  for loss in soft ; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3,3\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/dev \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/test \
      --input-per-spks 384 \
      --nj 12 \
      --epochs 24 \
      --resnet-size 18 \
      --embedding-size 128 \
      --kernel-size 3,3 \
      --avg-size 4 \
      --milestones 10,15,20 \
      --channels 64,128,256,256 \
      --check-path Data/checkpoint/LoResNet18/${dataset}/spect/${loss}_dp25 \
      --resume Data/checkpoint/LoResNet18/${dataset}/spect/${loss}_dp25/checkpoint_1.pth \
      --loss-type ${loss} \
      --lr 0.1 \
      --num-valid 2 \
      --gpu-id 1 \
      --dropout-p 0.25
  done

fi

if [ $stage -le 30 ]; then
  dataset=cnceleb
  model=LoResNet
  resnet_size=8
  for loss in soft ; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5,5\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/dev \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/test \
      --input-per-spks 224 \
      --feat-format npy \
      --nj 12 \
      --epochs 24 \
      --resnet-size ${resnet_size} \
      --embedding-size 128 \
      --avg-size 4 \
      --milestones 10,15,20 \
      --channels 64,128,256 \
      --check-path Data/checkpoint/LoResNet8/${dataset}/spect/${loss}_dp25 \
      --resume Data/checkpoint/LoResNet8/${dataset}/spect/${loss}_dp25/checkpoint_1.pth \
      --loss-type ${loss} \
      --lr 0.1 \
      --num-valid 2 \
      --gpu-id 0 \
      --dropout-p 0.25
  done

fi

stage=50
if [ $stage -le 40 ]; then
  datasets=all_army
  model=LoResNet
  resnet_size=8
  for loss in soft; do
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/storage/yangwenhao/project/lstm_speaker_verification/data/all_army/spect/dev \
      --test-dir /home/storage/yangwenhao/project/lstm_speaker_verification/data/all_army/spect/test \
      --feat-format npy \
      --resnet-size 8 \
      --nj 10 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss} \
      --resume Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss}/checkpoint_1.pth \
      --channels 64,128,256 \
      --embedding-size 128 \
      --input-per-spks 192 \
      --num-valid 1 \
      --alpha 12 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi