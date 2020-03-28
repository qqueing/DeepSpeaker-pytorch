# Speaker Verification Systems - Pytorch Implementation 

This project was stared from the [qqueing/DeepSpeaker-pytorch](https://github.com/qqueing/DeepSpeaker-pytorch). 

## Datasets

- Development:
> Voxceleb1
> Voxceleb2

- Augmentation:
> MUSAN
> RIRS

- Test:
> SITW
> Librispeech
> TIMIT

## Implemented Verification Systems

### TDNN

### SuResCNN

### LSTM

### ResNet

## To do list
Work accomplished so far:

- [x] Models implementation
- [x] Data pipeline implementation - "Voxceleb"
- [x] Project structure cleanup.
- [ ] Trained simple ResNet10 with softmax+triplet loss for pre-training 10 batch and triplet loss for 18 epoch , resulted in accuracy ???

## Timeline
- [x] Extract x-vectors from trained Neural Network in 20190626
- [ ] Code cleanup
- [ ] Modified preprocessing
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


The following is part of the original description:

>This is a slightly modified pytorch implementation of the model(modified Resnet + triplet loss) presented by Baidu Research in [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf).
>This code was tested using Voxceleb database. [Voxceleb database paper](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf) shows shows 7.8% EER using CNN. But in my code can't reach that point.
This code contained a lot of editable point such as preprocessing, model, scoring(length nomalization and file processing) and etc.
>I hope this code helps researcher reach higher score.
>Also, use the part of code:
> - [liorshk's git repository](https://github.com/liorshk/facenet_pytorch)
>   - Baseline code - Facenet pytorch implimetation
> - [hbredin's git repository](https://github.com/hbredin/pyannote-db-voxceleb)
>   - Voxceleb Database reader
>
> ### Features
> - In test, length normalization
> - This means extracting many input from single wave and averaging. This makes the results slightly better.
> - In training, except pandas and preloading list. 
> - This makes different training accuracy each epoch, but it does not matter.
>
> ### Authors
> qqueing@gmail.com( or kindsinu@naver.com)






