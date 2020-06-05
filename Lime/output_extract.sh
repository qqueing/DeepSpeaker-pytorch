#!/usr/bin/env bash

stage=60
waited=0
while [ `ps 15414 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

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

  python Lime/output_extract.py \
    --model LoResNet10 \
    --start-epochs 24 \
    --epochs 24 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --loss-type soft \
    --check-path Data/checkpoint/LoResNet10/spect/soft_wcmvn \
    --extract-path Data/gradient/LoResNet10/spect/soft_wcmvn \
    --dropout-p 0.25 \
    --gpu-id 1 \
    --embedding-size 128 \
    --sample-utt 5000

  for loss in amsoft center ; do
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

#stage=2
if [ $stage -le 2 ]; then
  model=ExResNet34
  datasets=vox1
#  feat=fb64_wcmvn
#  loss=soft
#  python Lime/output_extract.py \
#      --model ${model} \
#      --start-epochs 30 \
#      --epochs 30 \
#      --resnet-size 34 \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb64_wcmvn \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb64_wcmvn \
#      --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
#      --loss-type ${loss} \
#      --stride 1 \
#      --remove-vad \
#      --kernel-size 3,3 \
#      --check-path Data/checkpoint/ExResNet34/vox1/fb64_wcmvn/soft_var \
#      --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_var \
#      --dropout-p 0.0 \
#      --gpu-id 0 \
#      --embedding-size 128 \
#      --sample-utt 10000

  feat=fb64_wcmvn
  loss=soft
  python Lime/output_extract.py \
      --model ExResNet34 \
      --start-epochs 30 \
      --epochs 30 \
      --resnet-size 34 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/dev_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/test_noc \
      --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
      --loss-type ${loss} \
      --stride 1 \
      --remove-vad \
      --kernel-size 3,3 \
      --check-path Data/checkpoint/ExResNet/spect/soft \
      --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_kaldi \
      --dropout-p 0.0 \
      --gpu-id 1 \
      --time-dim 1 \
      --avg-size 1 \
      --embedding-size 128 \
      --sample-utt 5000
fi

#stage=100
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

#stage=300

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
    --sample-utt 10000 \
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
  feat=spect_noc
  loss=soft

#  python Lime/output_extract.py \
#    --model LoResNet10 \
#    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
#    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
#    --start-epochs 15 \
#    --check-path Data/checkpoint/LoResNet10/${datasets}/${feat}/${loss} \
#    --epochs 15 \
#    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
#    --sample-utt 4000 \
#    --embedding-size 128 \
#    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss} \
#    --model ${model} \
#    --channels 4,32,128 \
#    --dropout-p 0.25

#  python Lime/output_extract.py \
#    --model LoResNet10 \
#    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
#    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
#    --start-epochs 15 \
#    --check-path Data/checkpoint/LoResNet10/${datasets}/${feat}/${loss}_var \
#    --epochs 15 \
#    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
#    --sample-utt 4000 \
#    --embedding-size 128 \
#    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss}_var \
#    --model ${model} \
#    --channels 4,32,128 \
#    --dropout-p 0.25
  python Lime/output_extract.py \
    --model LoResNet10 \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_noc \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_noc \
    --start-epochs 15 \
    --check-path Data/checkpoint/LoResNet10/${datasets}/${feat}/${loss} \
    --epochs 15 \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --sample-utt 4000 \
    --alpha 9.8 \
    --embedding-size 128 \
    --extract-path Data/gradient/${model}/${datasets}/${feat}/${loss} \
    --model ${model} \
    --channels 4,16,64 \
    --dropout-p 0.25
fi

if [ $stage -le 40 ]; then
  model=TDNN
  feat=fb40_wcmvn
    for loss in soft ; do
      echo -e "\033[31m==> Loss type: ${loss} \033[0m"
      python Lime/output_extract.py \
        --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb40_wcmvn \
        --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb40_wcmvn \
        --nj 14 \
        --start-epochs 18 \
        --model ${model} \
        --embedding-size 128 \
        --sample-utt 5000 \
        --feat-dim 40 \
        --remove-vad \
        --check-path Data/checkpoint/TDNN/fb40_wcmvn/soft \
        --extract-path Data/gradient/TDNN/fb40_wcmvn/soft \
        --loss-type soft \
        --gpu-id 0
    done
fi

if [ $stage -le 50 ]; then
  model=SiResNet34
  feat=fb40_wcmvn
    for loss in soft ; do
      echo -e "\033[31m==> Loss type: ${loss} \033[0m"
      python Lime/output_extract.py \
        --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb64 \
        --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb64 \
        --nj 14 \
        --start-epochs 21 \
        --epochs 21 \
        --model ${model} \
        --embedding-size 128 \
        --sample-utt 5000 \
        --feat-dim 64 \
        --kernel-size 3,3 \
        --stride 1 \
        --input-length fix \
        --remove-vad \
        --mvnorm \
        --check-path Data/checkpoint/SiResNet34/vox1/fb64_cmvn/soft \
        --extract-path Data/gradient/SiResNet34/vox1/fb64_cmvn/soft \
        --loss-type soft \
        --gpu-id 1
    done
fi

if [ $stage -le 60 ]; then
  model=LoResNet10
  feat=spect
  dataset=cnceleb
  for loss in soft ; do
    echo -e "\033[31m==> Loss type: ${loss} \033[0m"
    python Lime/output_extract.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/dev \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/eval \
      --nj 14 \
      --start-epochs 24 \
      --epochs 24 \
      --model ${model} \
      --embedding-size 128 \
      --sample-utt 2500 \
      --feat-dim 161 \
      --kernel-size 3,3 \
      --channels 64,128,256,256 \
      --resnet-size 18 \
      --check-path Data/checkpoint/LoResNet18/${dataset}/spect/${loss}_dp25 \
      --extract-path Data/gradient/LoResNet18/${dataset}/spect/${loss}_dp25 \
      --loss-type soft \
      --gpu-id 1
  done
fi