#!/bin/bash
# Yangwenhao 2019-12-16 20:27
train_cmd="Vector_Score/run.pl --mem 4G"
logdir=Log/PLDA


$train_cmd $logdir/ivector-mean.log \
    ivector-mean scp:Data/checkpoint/ResNet10/Fb_No/xvector.scp Data/checkpoint/ResNet10/Fb_No/mean.vec || exit 1;

$train_cmd $logdir/ivector-compute-lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=200  "ark:ivector-subtract-global-mean scp:Data/checkpoint/ResNet10/Fb_No/xvector.scp ark:- |" ark:Data/checkpoint/ResNet10/Fb_No/utt2spk Data/checkpoint/ResNet10/Fb_No/transform.mat || exit 1;

Vector_Score/utt2spk_to_spk2utt.pl ~/mydataset/checkpoint/ResNet10/Fb_No/new_utt2spk > ~/mydataset/checkpoint/ResNet10/Fb_No/new_spk2utt

$train_cmd $logdir/ivector-compute-plda.log \
    ivector-compute-plda ark:Data/checkpoint/ResNet10/Fb_No/new_spk2utt "ark:ivector-subtract-global-mean scp:Data/checkpoint/ResNet10/Fb_No/new_xvector.scp ark:- | transform-vec Data/checkpoint/ResNet10/Fb_No/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" Data/checkpoint/ResNet10/Fb_No/plda || exit 1;

$train_cmd $logdir/ivector-plda-scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 Data/checkpoint/ResNet10/Fb_No/plda - |" \
    "ark:ivector-subtract-global-mean Data/checkpoint/ResNet10/Fb_No/mean.vec scp:Data/checkpoint/ResNet10/Fb_No/new_test_xvector.scp ark:- | transform-vec Data/checkpoint/ResNet10/Fb_No/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean Data/checkpoint/ResNet10/Fb_No/mean.vec scp:Data/checkpoint/ResNet10/Fb_No/new_test_xvector.scp ark:- | transform-vec Data/checkpoint/ResNet10/Fb_No/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat 'Data/checkpoint/ResNet10/Fb_No/trails' | cut -d\  --fields=1,2 |" Data/checkpoint/ResNet10/Fb_No/scores_voxceleb1_test || exit 1;

trails=~/mydataset/checkpoint/ResNet10/Fb_No/trails
test_score=~/mydataset/checkpoint/ResNet10/Fb_No/scores_voxceleb1_test
eer=`compute-eer <(Vector_Score/prepare_for_eer.py $trails $test_score) 2> /dev/null`
mindcf1=`Vector_Score/compute_min_dcf.py --p-target 0.01 $test_score $trails 2> /dev/null`
mindcf2=`Vector_Score/compute_min_dcf.py --p-target 0.001 $test_score $trails 2> /dev/null`

echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"
