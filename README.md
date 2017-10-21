# Deep Speaker from Baidu Research -  Pytorch Implementation 

This is a slightly modified pytorch implementation of the model(modified Resnet + triplet loss) presented by Baidu Research in [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf).

This code was tested using Voxceleb database. [Voxceleb database paper](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf) shows shows 7.8% EER using CNN. But in my code can't reach that point.
This code contained a lot of editable point such as preprocessing, model, scoring(length nomalization and file processing) and etc.

I hope this code helps researcher reach higher score.


## Credits
Original paper:
- Baidu Research paper:
```
@article{DBLP:journals/corr/LiMJLZLCKZ17,
  author    = {Chao Li and Xiaokong Ma and Bing Jiang and Xiangang Li and Xuewei Zhang and Xiao Liu and Ying Cao and Ajay Kannan and Zhenyao Zhu},
  title     = {Deep Speaker: an End-to-End Neural Speaker Embedding System},
  journal   = {CoRR},
  volume    = {abs/1705.02304},
  year      = {2017},
  url       = {http://arxiv.org/abs/1705.02304},
  timestamp = {Wed, 07 Jun 2017 14:41:04 +0200},
  biburl    = {http://dblp.org/rec/bib/journals/corr/LiMJLZLCKZ17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

Also, use the part of code:
- [liorshk's git repository](https://github.com/liorshk/facenet_pytorch)
   - Baseline code - Facenet pytorch implimetation
- [hbredin's git repository](https://github.com/hbredin/pyannote-db-voxceleb)
   - Voxceleb Database reader

## Features
Work accomplished so far:
- [x] Model implementation
- [x] Data pipeline implementation - "Voxceleb"(Please note:Pytorch dataloader is so weak(High-load preprocessing and many thread))
- [ ] Code cleanup
- [ ] Modified preprocessing
- [ ] Modified model
- [ ] Modified scoring



## Authors
qqueing@gmail.com( or kindsinu@naver.com)

