# Speaker Verification Systems - Pytorch Implementation 

This project was stared from the [qqueing/DeepSpeaker-pytorch](https://github.com/qqueing/DeepSpeaker-pytorch). 
And it will be my main work.

## Datasets

- Development:
> Voxceleb1
> Voxceleb2
> CNCELEB
> Aishell-2

- Augmentation:
> MUSAN
> RIRS
> RADIO NOISE

- Test:
> SITW
> Librispeech
> TIMIT

## Pre-Processing

- Resample

- Butter Bandpass Filtering

- Augmentation

- [ ] LMS Filtering


## Implemented DNN Verification Systems

- TDNN
The newest TDNN is  implemented from 'https://github.com/cvqluu/TDNN/blob/master/tdnn.py'

- ResCNN

- LSTM & Attention-based LSTM

Input 40-dimensional MFCC.

- ResNet

ResNet34 with Fbank64.

## Implemented Loss Type

- A-Softmax

- AM-Softmax

- Center Loss

## Implemented Pooling Type

- Self-Attention

- Statistic Pooling

- Attention Statistic Pooling


## To do list
Work accomplished so far:

- [x] Models implementation
- [x] Data pipeline implementation - "Voxceleb"
- [x] Project structure cleanup.
- [ ] Trained simple ResNet10 with softmax+triplet loss for pre-training 10 batch and triplet loss for 18 epoch , resulted in accuracy ???

## Timeline
- [x] Extract x-vectors from trained Neural Network in 20190626
- [x] Code cleanup (factory model creation) 20200725
- [x] Modified preprocessing
- [x] Modified model for ResNet34,50,101 in 20190625
- [x] Added cosine distance in Triplet Loss(The previous distance is l2) in 20190703
- [ ] Adding scoring for identification
- [ ] Fork plda method for classification in python from: https://github.com/RaviSoji/plda/blob/master/plda/

## Performance

|Stage|Resnet Model|epoch|Loss Type|Loss value|Accuracy on Train/Test|
|:--------:|:------------:|:---:|:--------------:|:--------------:|:------------:|
|1| Resnet-10    |1:22 |Triplet | 6.6420:0.0113 | 0.8553/0.8431  | 
| | ResNet-34    |1:8  |Triplet | 8.0285:0.0301 | 0.8360/0.8302  |
|2| ...          |..:..|        |

### Reference:  
> [1] Cai, Weicheng, Jinkun Chen, and Ming Li. "Analysis of Length Normalization in End-to-End Speaker Verification System.." conference of the international speech communication association (2018): 3618-3622.
>
> [2] ...







