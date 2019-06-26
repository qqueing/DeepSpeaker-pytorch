# Deep Speaker from Baidu Research -  Pytorch Implementation 

This project is forked from the [qqueing/DeepSpeaker-pytorch](https://github.com/qqueing/DeepSpeaker-pytorch). Parts of code was modified for the server to process. And some module will be added in the future.

## This will be my To do list
Work accomplished so far:
- [x] Model implementation
- [x] Data pipeline implementation - "Voxceleb"(Please note:Pytorch dataloader is so weak(High-load preprocessing and many thread))
- [x] Project structure cleanup.
- [x] Trained simple ResNet10 with accuracy 0.84

|LayerName|NumofDup|OutputSize|
|:------------:|:----------:|:------------:|
| conv1        |1           |              | 
| conv2        |2           |              |
| conv3        |2           |              |
| conv4        |2           |              |
| conv5        |2           |              |
|avg_pool      |1           |              |
|fc            |1           |              |

- [x] Extract x-vectors from trained Neural Network in 20190626
- [ ] Code cleanup
- [ ] Modified preprocessing
- [x] Modified model for ResNet34,50,101 in 20190625
- [ ] Adding scoring for identification


The following is part of the original description:

>This is a slightly modified pytorch implementation of the model(modified Resnet + triplet loss) presented by Baidu Research in [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf).
>This code was tested using Voxceleb database. [Voxceleb database paper](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf) shows shows 7.8% EER using CNN. But in my code can't reach that point.
This code contained a lot of editable point such as preprocessing, model, scoring(length nomalization and file processing) and etc.
>I hope this code helps researcher reach higher score.
>Also, use the part of code:
>- [liorshk's git repository](https://github.com/liorshk/facenet_pytorch)
>   - Baseline code - Facenet pytorch implimetation
>- [hbredin's git repository](https://github.com/hbredin/pyannote-db-voxceleb)
>   - Voxceleb Database reader
>
>## Features
> - In test, length normalization
>   - This means extracting many input from single wave and averaging. This makes the results slightly better.
> - In training, except pandas and preloading list. 
>   - This makes different training accuracy each epoch, but it does not matter.
>
> ## Authors
> qqueing@gmail.com( or kindsinu@naver.com)






