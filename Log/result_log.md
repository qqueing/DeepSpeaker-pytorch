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

####20190922
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