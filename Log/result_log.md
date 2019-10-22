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
