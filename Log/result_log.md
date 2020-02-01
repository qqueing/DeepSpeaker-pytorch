####20190805:
Trained **ResCNN34** with softmax cross entropy for 42 epoch:

    => loading checkpoint Data/checkpoint/resnet34_devall/checkpoint_42.pth
    Test Epoch: 43 [113088/37720 (100%)]: : 590it [10:19,  1.11s/it]
    For cos_distance Test set: ERR: 0.22773065	Best ACC:0.49997349

    => loading checkpoint Data/checkpoint/resnet34_devall/checkpoint_42.pth
    Test Epoch: 43 [113088/37720 (100%)]: : 590it [09:15,  1.09s/it]
    For l2_distance Test set: ERR: 22.81018028%	Best ACC:0.77298515

    => loading checkpoint Data/checkpoint/resnet34_devall/checkpoint_50.pth
    Test Epoch: 51 [113088/37720 (100%)]: : 590it [09:02,  1.07s/it]
    For cos_distance Test set: ERR: 23.36691410%	Best ACC:0.49997349

    => loading checkpoint Data/checkpoint/resnet34_devall/checkpoint_50.pth
    Test Epoch: 51 [113088/37720 (100%)]: : 590it [10:39,  1.10s/it]
    For l2_distance Test set: ERR: 23.48356310%	Best ACC:0.76574761

    => loading checkpoint Data/checkpoint/resnet34_devall/checkpoint_55.pth
    Test Epoch: 56 [113088/37720 (100%)]: : 590it [23:06,  2.54s/it]
    For cos_distance Test set: ERR: 23.22905620%	Best ACC:0.49997349

    => loading checkpoint Data/checkpoint/resnet34_devall/checkpoint_55.pth
    Test Epoch: 56 [113088/37720 (100%)]: : 590it [22:35,  2.59s/it]
    For l2_distance Test set: ERR: 23.02757158%	Best ACC:0.77014846

####20190827
Trained **ResCNN10** with asoftmax loss for 45 epoch:

    => loading checkpoint Data/checkpoint/resnet10_asoftmax/deprecated/checkpoint_45.pth
    start epoch is : 46
    Test Epoch: 46 [113088/37720 (100%)]: : 590it [08:01,  1.42s/it]
    For cos_distance Test set: ERR: 9.47507953%	Best ACC:0.50037116

####20190922 vosceleb1
Trained **ResCNN10** with **softmax** for 10 epoch, and **triplet loss** for 15 epoch:
    
    Train Epoch:  19 [  818688/  819200 (100%)]	Loss: 0.009060: : 1600it [1:00:17,  2.22s/it]
    0it [00:00, ?it/s]For cos_distance verification:
      Train set: ERR: 0.09555664	Best Accuracy:0.50000000 
    
    Test Epoch: 19 [206976/37720 (100%)]: : 295it [03:08,  1.73it/s]
    0it [00:00, ?it/s]For cos_distance Test set: ERR: 0.17401909	Best ACC:0.50000000

    Train Epoch:  24 [  818688/  819200 (100%)]	Loss: 0.006504: : 1600it [1:02:24,  2.22s/it]
    0it [00:00, ?it/s]For cos_distance verification:
      Train set: ERR: 0.09517700	Best Accuracy:0.50000000 
    
    Test Epoch: 24 [206976/37720 (100%)]: : 295it [03:05,  1.75it/s]
    0it [00:00, ?it/s]For cos_distance Test set: ERR: 0.18011665	Best ACC:0.50000000 
    
####20190930 voxceleb1
Trained **ResCNN34**  with **asoftmax** for 30 epoch:
    
    Train Epoch:   7 [   39474/  148642 (100%)]	Loss: 67.098373 	Minibatch Accuracy: 0.000000%: : 1162it [41:31,  1.74s/it]
    0it [00:00, ?it/s]For ASoftmax Train set Accuracy:0.073331% 
    
    Test Epoch: 7 [113088/37720 (100%)]: : 590it [22:10,  1.32s/it]
    0it [00:00, ?it/s]For cos_distance Test set: ERR: 7.75715801%	Best ACC:0.50000000 
    
    Train Epoch:   8 [   39474/  148642 (100%)]	Loss: 67.098366 	Minibatch Accuracy: 0.000000%: : 1162it [19:27,  1.23it/s]
    0it [00:00, ?it/s]For ASoftmax Train set Accuracy:0.073331% 
    
    Test Epoch: 8 [113088/37720 (100%)]: : 590it [07:09,  1.62it/s]
    0it [00:00, ?it/s]For cos_distance Test set: ERR: 8.18133616%	Best ACC:0.50058324 
    
####20191010 tdnn vox2

    => loading checkpoint Data/checkpoint/tdnn/checkpoint_1.pth
    checkpoint file epoch is : 2
    Test Epoch: 2 [14136/37720 (100%)]: : 590it [47:45,  5.11it/s]
    For cos_distance Test set: ERR: 27.75185578%	Best ACC:0.50000000 
    
####20191020 vox1 superficialResCNN

start epoch is : 1, Current learning rate is 0.1. 
    
    Train Epoch:   1 [   39474/  148642 (100%)]	Loss: 5.610270  Minibatch Accuracy: 17.647059%: : 1162it [44:04,  3.53s/it]s/it]
    
    For epoch 1: ASoftmax Train set Accuracy:3.151195%, and Average loss is 6.5809806255. 
    
    Test Epoch: 1 [113088/37720 (100%)]: : 590it [23:33,  3.21s/it]
    For cos_distance, Test set ERR: 26.75503712%	Best ACC:0.50000000 
    
    Train Epoch:  24 [   39474/  148642 (100%)]	Loss: 3.011633 	Minibatch Accuracy: 79.411765%: : 1162it [39:01,  2.16s/it]
    0it [00:00, ?it/s]
    For epoch 24: ASoftmax Train set Accuracy:83.524845%, and Average loss is 2.50831409494. 
    
    Test Epoch: 24 [100448/37720 (99%)]: : 74it [13:23,  8.96s/it]
    0it [00:00, ?it/s]For cos_distance, Test set ERR: 10.50371156%	Best ACC:0.50000000 
    
    Train Epoch:  47 [   39474/  148642 (100%)]	Loss: 1.890628 	Minibatch Accuracy: 94.117647%: : 1162it [41:57,  1.91s/it]
    0it [00:00, ?it/s]
    For epoch 47: ASoftmax Train set Accuracy:88.743424%, and Average loss is 1.99239653729. 
    
    Test Epoch: 47 [100448/37720 (99%)]: : 74it [13:16,  8.88s/it]
    For cos_distance, Test set ERR: 10.83775186%	Best ACC:0.50000000
    
    
Current learning rate is 0.01. 

    Train Epoch:  64 [  140800/  148642 ( 95%)]	Loss: 1.805255 	Minibatch Accuracy: 88.281250%: : 1162it [27:04,  1.15s/it]
    0it [00:00, ?it/s]
    For epoch 64: ASoftmax Train set Accuracy:89.590425%, and Average loss is 1.88088606239. 
    
    Test Epoch: 64 [0/37720 (0%)]: : 74it [15:46,  9.69s/it]
    0it [00:00, ?it/s]For cos_distance, Test set ERR is 10.95970308 when threshold is 0.82833866775	And test accuracy could be 89.12%.
    
     Current learning rate is 0.01. 
    
    Train Epoch:  65 [  140800/  148642 ( 95%)]	Loss: 1.825498 	Minibatch Accuracy: 88.281250%: : 1162it [21:55,  1.19it/s]
    0it [00:00, ?it/s]
    For epoch 65: ASoftmax Train set Accuracy:89.560824%, and Average loss is 1.88414668155. 

    
####20191022 tdnn vox1 test
    
    Test Epoch: 26 [103488/37720 (100%)]: : 295it [1:04:06, 14.53s/it]
    For cos_distance, Test set ERR is 19.46447508 when threshold is 0.40243518.	And test accuracy could be 80.55%.

    => loading checkpoint Data/checkpoint/tdnn_vox1/checkpoint_45.pth
    Test Epoch: 46 [103488/37720 (100%)]: : 295it [36:46,  7.07s/it]
    For cos_distance, Test set ERR is 19.04029692 when threshold is 0.39846855.	And test accuracy could be 81.02%.
    
####20191022 res10 softmax test
The model was trained in 20191013.

    => loading checkpoint Data/checkpoint/resnet10_softmax/checkpoint_8.pth
    Test Epoch: 9 [103488/37720 (100%)]: : 295it [03:05,  1.85s/it]
    For cos_distance, Test set ERR is 23.42523860 when threshold is 0.71897262.	And test accuracy could be 76.63%.
    
20191023, In epoch 9, change the loss to asoftmax and the result:

    Train Epoch:   9 [   39474/  148642 (100%)]	Loss: 6.473408 	Minibatch Accuracy: 76.470588%: : 1162it [34:28,  3.02s/it]
    0it [00:00, ?it/s]
    For epoch 9: ASoftmax Train set Accuracy:76.316923%, and Average loss is 6.42582087673. 
    
    Test Epoch: 9 [113088/37720 (100%)]: : 590it [22:44,  3.66s/it]
    0it [00:00, ?it/s]For cos_distance, Test set ERR is 10.92258749 when threshold is 0.676290810108. And test accuracy could be 89.13%.
        
    Train Epoch:  18 [   39474/  148642 (100%)]	Loss: 6.570085 	Minibatch Accuracy: 70.588235%: : 1162it [37:33,  1.57s/it]
    0it [00:00, ?it/s]
    For epoch 18: ASoftmax Train set Accuracy:74.065876%, and Average loss is 6.55523387331. 
    
    Test Epoch: 18 [113088/37720 (100%)]: : 590it [23:06,  1.85s/it]
    0it [00:00, ?it/s]For cos_distance, Test set ERR is 11.68610817 when threshold is 0.649993985891. And test accuracy could be 88.33%.
    
####20191022 res34 asoftmax vox1

    Train Epoch:  13 [  147200/  148642 ( 99%)]	Loss: 6.427779 	Minibatch Accuracy: 2.343750%: : 1162it [1:00:25,  2.36s/it]
    0it [00:00, ?it/s]For ASoftmax Res34 Train set Accuracy:1.367043%, and average loss 6.423520.
    
    Test Epoch: 13 [35200/37720 (93%)]: : 590it [50:59,  4.37s/it]
    0it [00:00, ?it/s]For cos_distance, Test set ERR is 21.67020148 when threshold is 0.327157497406	And est accuracy could be 79.14%.
    
    
    Train Epoch:  18 [  147200/  148642 ( 99%)]	Loss: 6.390326 	Minibatch Accuracy: 1.562500%: : 1162it [47:59,  2.14s/it]
    0it [00:00, ?it/s]For ASoftmax Res34 Train set Accuracy:1.513704%, and average loss 6.393960.
    
    Test Epoch: 18 [35200/37720 (93%)]: : 590it [49:45,  3.90s/it]
    0it [00:00, ?it/s]For cos_distance, Test set ERR is 22.90562036 when threshold is 0.339121937752	And est accuracy could be 78.32%.

#### 20191119 res10
    => no checkpoint found at Data/checkpoint/triplet/res10_soft/checkpoint_17.pth
    start epoch is : 1
    Train Epoch:   1 [  818560/  819200 (100%)]	Batch Loss: 0.1604103   Batch Accuracy: 96.8750%: : 12800it [4:47:00,  1.15s/it]
    For l2_distance verification:	Train set Accuracy: 0.71946289
    
    Test Epoch: 1 [37120/37720 (98%)]: : 295it [03:59,  1.36s/it]
    For l2_distance, Test set ERR is 21.37327678%, when threshold is 18.622446060180664. Best Test accuracy is 78.66%.
    
    Train Epoch:   2 [  818560/  819200 (100%)]	Batch Loss: 0.063731	Batch Accuracy: 98.9583%: : 12800it [4:08:14,  1.15s/it] 
    For l2_distance verification:	Train set Accuracy: 0.98223836
    
    Test Epoch: 2 [37120/37720 (98%)]: : 295it [02:58,  1.82it/s]
    For l2_distance, Test set ERR is 21.11876988%, when threshold is 20.90996551513672. Best Test accuracy is 78.99%.
    
    Train Epoch:   3 [  818560/  819200 (100%)]	Batch Loss: 0.017586	Batch Accuracy: 99.4792%: : 12800it [4:08:11,  1.15s/it] 
    For l2_distance verification:	Train set Accuracy: 0.99589315
    
    Test Epoch: 3 [37120/37720 (98%)]: : 295it [02:58,  1.80it/s]
    For l2_distance, Test set ERR is 20.88016967%, when threshold is 21.78229331970215. Best Test accuracy is 79.18%.
    
#### 20191119 res10 crossentropy+ triplet loss
=> no checkpoint found at Data/checkpoint/triplet/res10_soft/checkpoint_3.pth
start epoch is : 1
Train Epoch:   1 [  203520/  204800 ( 99%)]	Batch Loss: 3.3184121   Batch Accuracy: 30.7292%: : 1600it [2:03:44,  3.79s/it]
For l2_distance verification:	Train set Accuracy: 0.08250488
Test Epoch for Classification:   1 [    5120/    6055 ( 83%)]	Batch Accuracy: 20.3125%: : 48it [01:08,  1.67s/it]
Test Epoch: 1 [37120/37720 (98%)]: : 295it [04:19,  1.49s/it]
For l2_distance, Test set verification ERR is 25.74231177%, when threshold is 9.752527236938477. Valid set classificaton accuracy is 21.02%.
