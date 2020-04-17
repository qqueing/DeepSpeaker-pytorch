    
###2020.02.28:

#####TDNN: Fbank40

sgd lr=0.001, epoch=[30, 45], dropout=0.2, embedding_dim=512

> Cross Entropy:

        tdnn    64 with 256 repeat per speaker


###2020.03.06:

#####LSTM: Fbank 40

adam lr=0.001, epoch=[45,90], dropout=0.1, embedding_dim=512
    
> Cross Entropy:
        
        lstm    64 with 256 repeat per speaker
        alstm   64 with 256 repeat per speaker
    
> Tuple 0.9 * ce + 0.1 * tuple:

        lstm    32*5 with 192 repeat per speaker
        alstm   32*5 with 192 repeat per speaker
        
#####SuResCNN10: Spectrogram 161

sgd lr=0.05, epoch=[10, 15, 20], dropout=0.0, embedding_dim=1024

- [x] Cross Entropy:
        
        SuResCNN10    64 with 160 repeat per speaker
        # alstm   64 with 256 repeat
    
> Angular 1-0.1x * ce + 0.1x * as:

        lstm    32*5 with 160 repeat per speaker
        # alstm   32*5 with 192 repeat

    => loading checkpoint Data/checkpoint/SuResCNN10/spect/kaldi_5wd/checkpoint_20.pth
    start epoch is : 20
    
    For Sitw Test ERR is 14.6667%, Threshold is 0.24861189723014832.
    For Sitw Test ERR is 13.5922%, Threshold is 0.2714540958404541.
    
###2020.03.28:

#####ExResNet34: Fbank 64


sgd lr=0.05, epoch=[15, 24, 32], dropout=0.0, embedding_dim=128

- [x] Cross Entropy:
        
        Voxceleb 1
        Voxceleb 1 Augmented

###2020.04.06
##### train resnet with greater filter kernels
