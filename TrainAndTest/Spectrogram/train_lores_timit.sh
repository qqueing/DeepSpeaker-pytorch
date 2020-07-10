#!/usr/bin/env bash

#stage=3
stage=15

waited=0
while [ `ps 27212 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 0 ]; then
  datasets=timit
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
#    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
#      --model LoResNet10 \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/dev_wcmvn \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_wcmvn \
#      --nj 14 \
#      --epochs 14 \
#      --lr 0.1 \
#      --milestones 7,11 \
#      --check-path Data/checkpoint/LoResNet10/${datasets}/spect_wcmvn/${loss} \
#      --resume Data/checkpoint/LoResNet10/${datasets}/spect_wcmvn/${loss}/checkpoint_1.pth \
#      --channels 4,16,64 \
#      --statis-pooling \
#      --embedding-size 128 \
#      --input-per-spks 256 \
#      --num-valid 1 \
#      --weight-decay 0.001 \
#      --dropout-p 0.25 \
#      --loss-type ${loss}

    python TrainAndTest/Spectrogram/train_lores10_var.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_noc \
      --nj 8 \
      --epochs 12 \
      --lr 0.1 \
      --milestones 6,10 \
      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss}_var \
      --resume Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss}_var/checkpoint_7.pth \
      --channels 4,16,64 \
      --statis-pooling \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --weight-decay 0.001 \
      --alpha 10.8 \
      --dropout-p 0.5 \
      --gpu-id 1 \
      --loss-type ${loss}
  done
fi

#stage=60
if [ $stage -le 6 ]; then
  datasets=libri
  model=LoResNet
  resnet_size=8
#  --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
#  --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
  for loss in soft asoft ; do
    echo -e "\n\033[1;4;31m Training ${model} with ${loss} in ${datasets}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_wcmvn \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/spect_wcmvn/${loss} \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/spect_wcmvn/${loss}/checkpoint_1.pth \
      --kernel-size 5,5 \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --alpha 10 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.1 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --loss-type ${loss}

#    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
#      --model ${model} \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_wcmvn \
#      --resnet-size 8 \
#      --nj 12 \
#      --epochs 15 \
#      --lr 0.1 \
#      --milestones 7,11 \
#      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_wcmvn/${loss}_norm_fc \
#      --resume Data/checkpoint/LoResNet8/${datasets}/spect_wcmvn/${loss}_norm_fc/checkpoint_1.pth \
#      --channels 4,16,64 \
#      --embedding-size 128 \
#      --input-per-spks 256 \
#      --num-valid 1 \
#      --alpha 10 \
#      --margin 0.4 \
#      --s 30 \
#      --m 3 \
#      --loss-ratio 0.05 \
#      --weight-decay 0.001 \
#      --dropout-p 0.25 \
#      --inst-norm \
#      --gpu-id 0 \
#      --loss-type ${loss}
  done
fi

if [ $stage -le 10 ]; then
  datasets=army
  model=LoResNet
  resnet_size=8
  for loss in soft; do
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/army/dev_10k \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/army/test \
      --resnet-size 8 \
      --nj 12 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss} \
      --resume Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss}/checkpoint_1.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --alpha 10 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --inst-norm \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi

if [ $stage -le 15 ]; then
  datasets=army
  model=LoResNet
  resnet_size=8
  for loss in soft; do
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/storage/yangwenhao/project/lstm_speaker_verification/data/army/spect/dev \
      --test-dir /home/storage/yangwenhao/project/lstm_speaker_verification/data/army/spect/test \
      --feat-format npy \
      --resnet-size 8 \
      --nj 10 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss}_norm \
      --resume Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss}_norm/checkpoint_1.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 192 \
      --num-valid 1 \
      --alpha 10 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --inst-norm \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi
exit 0;
