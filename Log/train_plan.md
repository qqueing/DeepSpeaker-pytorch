## 20191012 

### Asoftmax ResCNN Vox1
Trained with ResCNN with softmax+asoftmax, learning rate is 0.1 with adagrad optimizer. Input is 32 frame for each utterance.

Logging file is Log/asoftmax_res10/res10_ssa_vox1_1012.txt

Result: 
* Loss decreased with too small pace(~6.4).
* Test set EER is 24%.


## 20191013 started.
### Softmax TDNN vox1 
Training configure: learning rate is learning rate is 0.1 with adagrad optimizer. Input is 300 frame for each utterance.


### Softmax ResCNN vox1: 20191015 completed!
Training configure: learning rate is learning rate is 0.1 with adagrad optimizer. Input is 300 frame for each utterance.

The accuracy is up to 99%. However the eer on test set of vex1 decreased to 14%, then the model seemed to overfit.

### Using ResCNN that was trained with softmax for 8 epoch, then trained with asoftmax:
In epoch 8, eer on test set is almost 12%.

### Using simple ResNet train with softmax for 30 epoch

Parsed options: {'ckp_dir': 'Data/checkpoint/ResNet10/soft', 'log_dir': 'data/pytorch_speaker_logs', 'dataroot': '/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/fbank64', 'epochs': 37, 'makespec': False, 'lr_decay': 0, 'start_epoch': 1, 'log_interval': 15, 'n_triplets': 100000, 'batch_size': 64, 'lr': 0.05, 'wd': 0.001, 'embedding_size': 512, 'gpu_id': '1', 'resume': 'Data/checkpoint/ResNet10/soft/checkpoint_20.pth', 'cos_sim': True, 'margin': 0.1, 'test_input_per_file': 1, 'test_batch_size': 64, 'test_dataroot': '/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/fbank64', 'seed': 3, 'optimizer': 'adagrad', 'min_softmax_epoch': 20, 'cuda': True, 'no_cuda': False, 'makemfb': False, 'loss_ratio': 2.0, 'acoustic_feature': 'fbank', 'test_pairs_path': 'Data/dataset/ver_list.txt'}

Parsed options: {'n_triplets': 100000, 'test_dataroot': '/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/Fbank64_Norm', 'log_interval': 15, 'optimizer': 'adagrad', 'cuda': True, 'batch_size': 64, 'log_dir': 'data/pytorch_speaker_logs', 'cos_sim': True, 'no_cuda': False, 'embedding_size': 512, 'gpu_id': '2', 'acoustic_feature': 'fbank', 'test_input_per_file': 1, 'epochs': 35, 'dataroot': '/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/Fbank64_Norm', 'makespec': False, 'lr': 0.05, 'ckp_dir': 'Data/checkpoint/ResNet10/Fbank64_Norm', 'min_softmax_epoch': 20, 'lr_decay': 0, 'test_pairs_path': 'Data/dataset/ver_list.txt', 'loss_ratio': 2.0, 'start_epoch': 1, 'resume': 'Data/checkpoint/ResNet10/Fbank64_Norm/checkpoint_20.pth', 'makemfb': False, 'wd': 0.001, 'test_batch_size': 64, 'margin': 0.1, 'seed': 3}


### 20191216 For SiResNet34 
    the eer should be 5.01.