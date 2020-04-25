#!/usr/bin/env bash

waited=0
while [ `ps 151321 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

stage=0
if [ $stage -le 0 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_resnet20_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257_wcmvn \
      --embedding-size 128 \
      --batch-size 32 \
      --test-batch-size 2 \
      --accumulation-steps 2 \
      --nj 12 \
      --epochs 24 \
      --milestones 10,15,20 \
      --lr 0.1 \
      --veri-pairs 12800 \
      --check-path Data/checkpoint/ResNet20/spect_257_wcmvn/${loss}_dp0.5 \
      --resume Data/checkpoint/ResNet20/spect_257_wcmvn/${loss}_dp0.5/checkpoint_1.pth \
      --loss-type ${loss} \
      --dropout-p 0.5
  done
fi

#stage=2
if [ $stage -le 1 ]; then
  for loss in amsoft asoft ; do
    echo -e "\n\033[1;4;31m Finetuning with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_resnet20_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257 \
      --embedding-size 128 \
      --batch-size 32 \
      --test-batch-size 2 \
      --accumulation-steps 2 \
      --nj 12 \
      --lr 0.01 \
      --finetune \
      --epochs 16 \
      --milestones 8,12 \
      --veri-pairs 12800 \
      --check-path Data/checkpoint/ResNet20/spect_257/${loss}_dp0.5 \
      --resume Data/checkpoint/ResNet20/spect_257/soft_dp0.5/checkpoint_24.pth \
      --loss-type ${loss} \
      --m 4 \
      --margin 0.4 \
      --s 50 \
      --dropout-p 0.5
  done
fi

#if [ $stage -le 2 ]; then
##  for loss in amsoft asoft ; do
#  for loss in asoft ; do
#    echo -e "\n\033[1;4;31m Finetuning with ${loss}\033[0m\n"
#    python TrainAndTest/Spectrogram/train_resnet20_kaldi.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257 \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257 \
#      --embedding-size 128 \
#      --batch-size 32 \
#      --test-batch-size 2 \
#      --accumulation-steps 2 \
#      --nj 12 \
#      --epochs 15 \
#      --milestones 8,12 \
#      --lr 0.01 \
#      --veri-pairs 12800 \
#      --check-path Data/checkpoint/ResNet20/spect_257/${loss}_fine \
#      --resume Data/checkpoint/ResNet20/spect_257/soft_dp0.5/checkpoint_20.pth \
#      --loss-type ${loss} \
#      --m 3 \
#      --margin 0.3 \
#      --s 30 \
#      --dropout-p 0.5
#  done
#fi