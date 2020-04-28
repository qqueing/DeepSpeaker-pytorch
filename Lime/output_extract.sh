#!/usr/bin/env bash

stage=3
if [ $stage -le 0 ]; then
  for model in LoResNet10 ; do
    python Lime/output_extract.py \
      --model ${model} \
      --epochs 19 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
      --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --check-path /home/yangwenhao/local/project/DeepSpeaker-pytorch/Data/checkpoint/LoResNet10/spect/soft \
      --extract-path Lime/${model} \
      --dropout-p 0.5 \
      --sample-utt 500

  done
fi

if [ $stage -le 1 ]; then
#  for model in LoResNet10 ; do
#  python Lime/output_extract.py \
#    --model LoResNet10 \
#    --start-epochs 36 \
#    --epochs 36 \
#    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev \
#    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test \
#    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
#    --loss-type center \
#    --check-path /home/yangwenhao/local/project/DeepSpeaker-pytorch/Data/checkpoint/LoResNet10/spect_cmvn/center_dp25 \
#    --extract-path Data/gradient \
#    --dropout-p 0 \
#    --gpu-id 0 \
#    --embedding-size 1024 \
#    --sample-utt 2000

#  python Lime/output_extract.py \
#    --model LoResNet10 \
#    --start-epochs 24 \
#    --epochs 24 \
#    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
#    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
#    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
#    --loss-type soft \
#    --check-path Data/checkpoint/LoResNet10/spect/soft_wcmvn \
#    --extract-path Data/gradient/LoResNet10/spect/soft_wcmvn \
#    --dropout-p 0.25 \
#    --gpu-id 1 \
#    --embedding-size 128 \
#    --sample-utt 5000

  for loss in amsoft ; do
    python Lime/output_extract.py \
      --model LoResNet10 \
      --start-epochs 38 \
      --epochs 38 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
      --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --loss-type ${loss} \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_wcmvn \
      --extract-path Data/gradient/LoResNet10/spect/${loss}_wcmvn \
      --dropout-p 0.25 \
      --s 15 \
      --margin 0.35 \
      --gpu-id 1 \
      --embedding-size 128 \
      --sample-utt 5000
  done
fi

if [ $stage -le 2 ]; then
  model=ExResNet34
  datasets=vox1
  feat=fb64_noc
  loss=soft
  python Lime/output_extract.py \
      --model ${model} \
      --start-epochs 30 \
      --epochs 30 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/dev_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/test_noc \
      --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --loss-type ${loss} \
      --check-path Data/checkpoint/ExResNet/spect/soft \
      --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss} \
      --dropout-p 0.0 \
      --gpu-id 1 \
      --embedding-size 128 \
      --sample-utt 5000
fi

if [ $stage -le 3 ]; then
  model=ResNet20
  datasets=vox1
  feat=spect_256_wcmvn
  loss=soft
  python Lime/output_extract.py \
      --model ${model} \
      --start-epochs 24 \
      --epochs 24 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257_wcmvn \
      --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --loss-type ${loss} \
      --check-path Data/checkpoint/ResNet20/spect_257_wcmvn/soft_dp0.5 \
      --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_wcmvn \
      --dropout-p 0.5 \
      --gpu-id 1 \
      --embedding-size 128 \
      --sample-utt 5000
fi

stage=300

if [ $stage -le 20 ]; then
  model=LoResNet10
  datasets=timit
  feat=spect
  loss=soft

#  python Lime/output_extract.py \
#    --model LoResNet10 \
#    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
#    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_noc \
#    --start-epochs 15 \
#    --check-path Data/checkpoint/LoResNet10/timit_spect/soft_fix \
#    --epochs 15 \
#    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
#    --sample-utt 1500 \
#    --embedding-size 128 \
#    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_fix \
#    --model ${model} \
#    --channels 4,16,64 \
#    --dropout-p 0.25

  python Lime/output_extract.py \
    --model LoResNet10 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_noc \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_noc \
    --start-epochs 15 \
    --check-path Data/checkpoint/LoResNet10/timit_spect/soft_var \
    --epochs 15 \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --sample-utt 4000 \
    --embedding-size 128 \
    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_var \
    --model ${model} \
    --channels 4,16,64 \
    --dropout-p 0.25
fi

#stage=500

if [ $stage -le 30 ]; then
  model=LoResNet10
  datasets=libri
  feat=spect
  loss=soft

  python Lime/output_extract.py \
    --model LoResNet10 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
    --start-epochs 15 \
    --check-path Data/checkpoint/LoResNet10/${datasets}/${feat}/${loss} \
    --epochs 15 \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --sample-utt 4000 \
    --embedding-size 128 \
    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss} \
    --model ${model} \
    --channels 4,32,128 \
    --dropout-p 0.25

  python Lime/output_extract.py \
    --model LoResNet10 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
    --start-epochs 15 \
    --check-path Data/checkpoint/LoResNet10/${datasets}/${feat}/${loss}_var \
    --epochs 15 \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --sample-utt 4000 \
    --embedding-size 128 \
    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_var \
    --model ${model} \
    --channels 4,32,128 \
    --dropout-p 0.25
fi
