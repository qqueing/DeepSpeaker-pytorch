#!/bin/bash
# Yangwenhao 2019-12-16 20:27


model=SuResCNN10
train_cmd="Vector_Score/run.pl --mem 8G"
logdir=Log/PLDA/${model}

feat_dir=Data/checkpoint/${model}/soft/kaldi_feat
data_dir=Data/dataset/voxceleb1/kaldi_feat/voxceleb1_test

trials=$data_dir/trials
test_score=$feat_dir/scores_voxceleb1_test

$train_cmd $logdir/ivector-mean.log \
    ivector-mean scp:$feat_dir/train_xvector.scp $feat_dir/mean.vec || exit 1;

$train_cmd $logdir/ivector-compute-lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=200  "ark:ivector-subtract-global-mean scp:$feat_dir/train_xvector.scp ark:- |" ark:$feat_dir/utt2spk $feat_dir/transform.mat || exit 1;

Vector_Score/utt2spk_to_spk2utt.pl $feat_dir/utt2spk > $feat_dir/spk2utt

$train_cmd $logdir/ivector-compute-plda.log \
    ivector-compute-plda ark:$feat_dir/spk2utt "ark:ivector-subtract-global-mean scp:$feat_dir/train_xvector.scp ark:- | transform-vec $feat_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" $feat_dir/plda || exit 1;

$train_cmd $logdir/ivector-plda-scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $feat_dir/plda - |" \
    "ark:ivector-subtract-global-mean $feat_dir/mean.vec scp:$feat_dir/test_xvector.scp ark:- | transform-vec $feat_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $feat_dir/mean.vec scp:$feat_dir/test_xvector.scp ark:- | transform-vec $feat_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $feat_dir/scores_voxceleb1_test || exit 1;

eer=`compute-eer <(Vector_Score/prepare_for_eer.py $trials $test_score) 2> /dev/null`
mindcf1=`Vector_Score/compute_min_dcf.py --p-target 0.01 $test_score $trials 2> /dev/null`
mindcf2=`Vector_Score/compute_min_dcf.py --p-target 0.001 $test_score $trials 2> /dev/null`

echo "EER: $eer%"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"
