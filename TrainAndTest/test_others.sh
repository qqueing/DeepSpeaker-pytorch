#!/usr/bin/env bash

stage=6

if [ $stage -le 0 ]; then
  for loss in asoft soft ; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"

    python TrainAndTest/test_sitw.py \
      --nj 12 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss} \
      --veri-pairs 12800 \
      --loss-type ${loss} \
      --gpu-id 0 \
      --epochs 20
  done
fi

if [ $stage -le 5 ]; then
  model=LoResNet10
#  --resume Data/checkpoint/LoResNet10/spect/${loss}_dp25_128/checkpoint_24.pth \
#  for loss in soft ; do
  for loss in soft ; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    python TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/all_wcmvn \
      --nj 12 \
      --model ${model} \
      --embedding-size 128 \
      --resume Data/checkpoint/LoResNet10/spect/${loss}_wcmvn/checkpoint_24.pth \
      --xvector-dir Data/xvector/LoResNet10/spect/${loss}_wcmvn \
      --loss-type ${loss} \
      --trials trials \
      --num-valid 0 \
      --gpu-id 0
  done

#  for loss in center ; do
#    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
#    python TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
#      --nj 12 \
#      --model ${model} \
#      --resume Data/checkpoint/LoResNet10/spect_cmvn/${loss}_dp25/checkpoint_36.pth \
#      --loss-type ${loss} \
#      --num-valid 2 \
#      --gpu-id 1
#  done
fi

if [ $stage -le 6 ]; then
  model=LoResNet10
#  --resume Data/checkpoint/LoResNet10/spect/${loss}_dp25_128/checkpoint_24.pth \
#  for loss in soft ; do
  for loss in soft ; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
#    python TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
#      --nj 12 \
#      --model ${model} \
#      --channels 64,128,256,512 \
#      --resnet-size 10 \
#      --extract \
#      --kernel-size 3,3 \
#      --embedding-size 128 \
#      --resume Data/checkpoint/LoResNet10/spect/soft_dp05/checkpoint_36.pth \
#      --xvector-dir Data/xvector/LoResNet10/spect/soft_dp05 \
#      --loss-type ${loss} \
#      --trials trials.backup \
#      --num-valid 0 \
#      --gpu-id 0
    python TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
      --nj 12 \
      --model ${model} \
      --channels 64,128,256,256 \
      --resnet-size 18 \
      --extract \
      --kernel-size 3,3 \
      --embedding-size 128 \
      --resume Data/checkpoint/LoResNet18/spect/soft_dp25/checkpoint_24.pth \
      --xvector-dir Data/xvector/LoResNet18/spect/soft_dp05 \
      --loss-type ${loss} \
      --trials trials.backup \
      --num-valid 0 \
      --gpu-id 0
  done

#  model=LoResNet10
#  for loss in soft ; do
#    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
#    python TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
#      --nj 12 \
#      --model ${model} \
#      --channels 64,128,256 \
#      --resnet-size 8 \
#      --kernel-size 5,5 \
#      --embedding-size 128 \
#      --resume Data/checkpoint/LoResNet8/spect/soft_wcmvn/checkpoint_24.pth \
#      --extract \
#      --xvector-dir Data/xvector/LoResNet8/spect/soft_wcmvn \
#      --loss-type ${loss} \
#      --trials trials.backup \
#      --num-valid 0 \
#      --gpu-id 0
#  done

#  for loss in center ; do
#    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
#    python TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
#      --nj 12 \
#      --model ${model} \
#      --resume Data/checkpoint/LoResNet10/spect_cmvn/${loss}_dp25/checkpoint_36.pth \
#      --loss-type ${loss} \
#      --num-valid 2 \
#      --gpu-id 1
#  done
fi

#python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
#    --check-path Data/checkpoint/LoResNet10/spect/amsoft \
#    --resume Data/checkpoint/LoResNet10/spect/soft/checkpoint_20.pth \
#    --loss-type amsoft \
#    --lr 0.01 \
#    --epochs 10

stage=200
if [ $stage -le 15 ]; then
  model=TDNN
#  feat=fb40
#  for loss in soft ; do
#    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
#    python TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb40_no_sil \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb40_no_sil \
#      --nj 12 \
#      --model ${model} \
#      --embedding-size 128 \
#      --feat-dim 40 \
#      --resume Data/checkpoint/${model}/${feat}/${loss}/checkpoint_18.pth
#      --loss-type soft \
#      --num-valid 2 \
#      --gpu-id 1
#  done

  feat=fb40_wcmvn
  for loss in soft ; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    python TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb40_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb40_wcmvn \
      --nj 14 \
      --model ${model} \
      --embedding-size 128 \
      --feat-dim 40 \
      --remove-vad \
      --extract \
      --valid \
      --resume Data/checkpoint/TDNN/fb40_wcmvn/soft_fix/checkpoint_40.pth \
      --xvector-dir Data/xvectors/TDNN/fb40_wcmvn/soft_fix \
      --loss-type soft \
      --num-valid 2 \
      --gpu-id 1
  done

fi

#stage=200
if [ $stage -le 20 ]; then
  model=LoResNet10
  feat=spect
  datasets=libri
  for loss in soft ; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
#    python TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
#      --nj 12 \
#      --model ${model} \
#      --channels 4,32,128 \
#      --embedding-size 128 \
#      --resume Data/checkpoint/${model}/${datasets}/${feat}/${loss}/checkpoint_15.pth \
#      --loss-type soft \
#      --dropout-p 0.25 \
#      --num-valid 1 \
#      --gpu-id 1
#
#    python TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
#      --nj 12 \
#      --model ${model} \
#      --channels 4,32,128 \
#      --embedding-size 128 \
#      --resume Data/checkpoint/${model}/${datasets}/${feat}/${loss}_var/checkpoint_15.pth \
#      --loss-type soft \
#      --dropout-p 0.25 \
#      --num-valid 1 \
#      --gpu-id 1
#    python TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
#      --nj 12 \
#      --model ${model} \
#      --channels 4,32,128 \
#      --embedding-size 128 \
#      --alpha 9.8 \
#      --extract \
#      --resume Data/checkpoint/LoResNet10/libri/spect_noc/soft/checkpoint_15.pth \
#      --xvector-dir Data/xvectors/LoResNet10/libri/spect_noc/soft_128 \
#      --loss-type ${loss} \
#      --dropout-p 0.25 \
#      --num-valid 2 \
#      --gpu-id 1
      python TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
      --nj 12 \
      --model ${model} \
      --channels 4,32,128 \
      --embedding-size 128 \
      --alpha 9.8 \
      --extract \
      --resume Data/checkpoint/LoResNet10/libri/spect_noc/soft_fix_43/checkpoint_15.pth \
      --xvector-dir Data/xvectors/LoResNet10/libri/spect_noc/soft_128 \
      --loss-type ${loss} \
      --dropout-p 0.25 \
      --num-valid 2 \
      --gpu-id 1


#    python TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
#      --nj 12 \
#      --model ${model} \
#      --channels 4,16,64 \
#      --embedding-size 128 \
#      --resume Data/checkpoint/LoResNet10/libri/spect_noc/soft_var/checkpoint_15.pth \
#      --loss-type soft \
#      --dropout-p 0.25 \
#      --num-valid 2 \
#      --gpu-id 1
  done
fi

#stage=250
if [ $stage -le 25 ]; then
  model=LoResNet10
  feat=spect_wcmvn
  datasets=timit
  for loss in soft ; do
#    echo -e "\033[31m==> Loss type: ${loss} variance_fix length \033[0m"
#    python TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_noc \
#      --nj 12 \
#      --model ${model} \
#      --channels 4,16,64 \
#      --embedding-size 128 \
#      --resume Data/checkpoint/LoResNet10/timit_spect/soft_fix/checkpoint_15.pth \
#      --loss-type soft \
#      --dropout-p 0.25 \
#      --num-valid 2 \
#      --gpu-id 1

    echo -e "\033[31m==> Loss type: ${loss} variance_fix length \033[0m"
    python TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
      --nj 12 \
      --model ${model} \
      --xvector-dir Data/xvectors/LoResNet10/timit_spect/soft_var \
      --channels 4,16,64 \
      --embedding-size 128 \
      --resume Data/checkpoint/LoResNet10/timit_spect/soft_var/checkpoint_15.pth \
      --loss-type soft \
      --dropout-p 0.25 \
      --num-valid 2 \
      --gpu-id 1
  done
fi

#stage=100
if [ $stage -le 30 ]; then
  model=ResNet20
  feat=spect_wcmvn
  datasets=vox
  for loss in soft ; do
    echo -e "\033[31m==> Loss type: ${loss} fix length \033[0m"
    python TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257_wcmvn \
      --nj 12 \
      --model ${model} \
      --embedding-size 128 \
      --resume Data/checkpoint/ResNet20/spect_257_wcmvn/soft_dp0.5/checkpoint_24.pth \
      --loss-type soft \
      --dropout-p 0.5 \
      --num-valid 2 \
      --gpu-id 1
  done
fi

#stage=100
if [ $stage -le 40 ]; then
  model=ExResNet34
#  for loss in soft asoft ; do
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Test ${model} with ${loss} vox_wcmvn\033[0m\n"
    python -W ignore TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb64_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb64_wcmvn \
      --nj 12 \
      --epochs 30 \
      --model ExResNet34 \
      --remove-vad \
      --resnet-size 34 \
      --embedding-size 128 \
      --feat-dim 64 \
      --kernel-size 3,3 \
      --stride 1 \
      --time-dim 1 \
      --avg-size 1 \
      --resume Data/checkpoint/ExResNet34/vox1/fb64_wcmvn/soft_14/checkpoint_22.pth \
      --xvector-dir Data/xvectors/ExResNet34/vox1/fb64_wcmvn/soft_14 \
      --input-per-spks 192 \
      --num-valid 2 \
      --extract \
      --gpu-id 1 \
      --loss-type ${loss}

#    echo -e "\n\033[1;4;31m Test ${model} with ${loss} vox_noc \033[0m\n"
#    python -W ignore TrainAndTest/test_vox1.py \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/dev_noc \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/test_noc \
#      --nj 12 \
#      --epochs 30 \
#      --model ExResNet34 \
#      --remove-vad \
#      --resnet-size 34 \
#      --embedding-size 128 \
#      --feat-dim 64 \
#      --kernel-size 3,3 \
#      --stride 1 \
#      --avg-size 1 \
#      --resume Data/checkpoint/ExResNet34/vox1/fb64_wcmvn/soft_14/checkpoint_22.pth \
#      --input-per-spks 192 \
#      --time-dim 1 \
#      --extract \
#      --num-valid 2 \
#      --loss-type ${loss}
  done
fi

if [ $stage -le 50 ]; then
#  for loss in soft asoft ; do
  model=SiResNet34
  datasets=vox1
  feat=fb64_mvnorm
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/test_vox1.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb64 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb64 \
      --nj 14 \
      --epochs 40 \
      --model ${model} \
      --resnet-size 34 \
      --embedding-size 128 \
      --feat-dim 64 \
      --remove-vad \
      --extract \
      --valid \
      --kernel-size 3,3 \
      --stride 1 \
      --mvnorm \
      --input-length fix \
      --test-input-per-file 4 \
      --xvector-dir Data/xvectors/${model}/${datasets}/${feat}/${loss} \
      --resume Data/checkpoint/SiResNet34/vox1/fb64_cmvn/soft/checkpoint_21.pth  \
      --input-per-spks 192 \
      --gpu-id 1 \
      --num-valid 2 \
      --loss-type ${loss}
  done
fi