    
###20200306:

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

> Cross Entropy:
        
        lstm    64 with 160 repeat per speaker
        # alstm   64 with 256 repeat
    
> Angular 1-0.1x * ce + 0.1x * as:

        lstm    32*5 with 160 repeat per speaker
        # alstm   32*5 with 192 repeat
