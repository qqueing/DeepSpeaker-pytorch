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