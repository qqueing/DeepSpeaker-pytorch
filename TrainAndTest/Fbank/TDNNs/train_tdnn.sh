#!/usr/bin/env bash

stage=15
waited=0
while [ `ps 103374 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 0 ]; then
  model=ETDNN
  for loss in soft ; do
    python TrainAndTest/Fbank/TDNNs/train_etdnn_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb80/dev_kaldi \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb80/test_kaldi \
      --check-path Data/checkpoint/${model}/fbank80/soft \
      --resume Data/checkpoint/${model}/fbank80/soft/checkpoint_1.pth
      --epochs 20 \
      --milestones 10,15  \
      --feat-dim 80 \
      --embedding-size 256 \
      --num-valid 2 \
      --loss-type soft \
      --lr 0.01

  done
fi

#stage=1
if [ $stage -le 5 ]; then
  model=TDNN
  feat=fb40
  for loss in soft ; do
    python TrainAndTest/Fbank/TDNNs/train_tdnn_var.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb40 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb40 \
      --check-path Data/checkpoint/${model}/${feat}/${loss}_norm \
      --resume Data/checkpoint/${model}/${feat}/${loss}_norm/checkpoint_1.pth \
      --batch-size 64 \
      --remove-vad \
      --epochs 16 \
      --milestones 8,12  \
      --feat-dim 40 \
      --embedding-size 128 \
      --weight-decay 0.0005 \
      --num-valid 2 \
      --loss-type ${loss} \
      --input-per-spks 192 \
      --gpu-id 0 \
      --veri-pairs 9600 \
      --lr 0.01
  done
fi

#stage=100
if [ $stage -le 10 ]; then
  model=ASTDNN
  feat=fb40_wcmvn
  for loss in soft ; do
#    python TrainAndTest/Fbank/TDNNs/train_astdnn_kaldi.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb40_wcmvn \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb40_wcmvn \
#      --check-path Data/checkpoint/${model}/${feat}/${loss} \
#      --resume Data/checkpoint/${model}/${feat}/${loss}/checkpoint_1.pth \
#      --epochs 18 \
#      --batch-size 128 \
#      --milestones 9,14  \
#      --feat-dim 40 \
#      --embedding-size 128 \
#      --num-valid 2 \
#      --loss-type ${loss} \
#      --input-per-spks 240 \
#      --lr 0.01

    python TrainAndTest/Fbank/TDNNs/train_tdnn_var.py \
      --model ASTDNN \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb40_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb40_wcmvn \
      --check-path Data/checkpoint/${model}/${feat}/${loss}_svar \
      --resume Data/checkpoint/${model}/${feat}/${loss}_svar/checkpoint_1.pth \
      --epochs 18 \
      --batch-size 128 \
      --milestones 9,14  \
      --feat-dim 40 \
      --remove-vad \
      --embedding-size 512 \
      --num-valid 2 \
      --loss-type ${loss} \
      --input-per-spks 240 \
      --lr 0.01 \
      --gpu-id 1

    python TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb40_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb40_wcmvn \
      --nj 12 \
      --model ASTDNN \
      --embedding-size 512 \
      --feat-dim 40 \
      --remove-vad \
      --resume Data/checkpoint/${model}/${feat}/${loss}_svar/checkpoint_18.pth \
      --loss-type soft \
      --num-valid 2 \
      --gpu-id 1

    python Lime/output_extract.py \
    --model ASTDNN \
    --start-epochs 18 \
    --epochs 18 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb40_wcmvn \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb40_wcmvn \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --loss-type soft \
    --remove-vad \
    --check-path Data/checkpoint/${model}/${feat}/${loss}_svar \
    --extract-path Data/gradient/${model}/${feat}/${loss}_svar \
    --gpu-id 1 \
    --embedding-size 512 \
    --sample-utt 5000
  done
fi

#stage=1
if [ $stage -le 15 ]; then
  model=ETDNN
  feat=fb80
  for loss in soft ; do
    python TrainAndTest/Fbank/TDNNs/train_tdnn_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb80 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb80 \
      --check-path Data/checkpoint/${model}/${feat}/${loss} \
      --resume Data/checkpoint/${model}/${feat}/${loss}/checkpoint_1.pth \
      --batch-size 128 \
      --remove-vad \
      --epochs 20 \
      --milestones 10,14  \
      --feat-dim 80 \
      --embedding-size 128 \
      --weight-decay 0.0005 \
      --num-valid 2 \
      --loss-type ${loss} \
      --input-per-spks 224 \
      --gpu-id 1 \
      --veri-pairs 9600 \
      --lr 0.01
  done
fi

stage=100
if [ $stage -le 16 ]; then
  model=ETDNN
  feat=fb80
  for loss in amsoft center ; do
    python TrainAndTest/Fbank/TDNNs/train_tdnn_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb80 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb80 \
      --check-path Data/checkpoint/${model}/${feat}/${loss} \
      --resume Data/checkpoint/ETDNN/fb80/soft/checkpoint_20.pth \
      --batch-size 128 \
      --remove-vad \
      --epochs 30 \
      --finetune \
      --milestones 6  \
      --feat-dim 80 \
      --embedding-size 128 \
      --weight-decay 0.0005 \
      --num-valid 2 \
      --loss-type ${loss} \
      --m 4 \
      --margin 0.35 \
      --s 15 \
      --loss-ratio 0.01 \
      --input-per-spks 224 \
      --gpu-id 1 \
      --veri-pairs 9600 \
      --lr 0.001
  done
fi