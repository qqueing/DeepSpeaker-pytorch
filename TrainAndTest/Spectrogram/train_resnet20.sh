#!/usr/bin/env bash

waited=0
while [ `ps 160299 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

stage=0
if [ $stage -le 0 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_resnet20_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257 \
      --embedding-size 128 \
      --batch-size 32 \
      --test-batch-size 2 \
      --accumulation-steps 2 \
      --nj 12 \
      --epochs 20 \
      --milestones 10,15 \
      --veri-pairs 12800 \
      --check-path Data/checkpoint/ResNet20/spect_257/${loss} \
      --resume Data/checkpoint/ResNet20/spect_257/${loss}/checkpoint_1.pth \
      --loss-type ${loss}
  done
fi