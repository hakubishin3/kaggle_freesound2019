## Features

I extract log-mel feature, the delta and accelerate of log-mel are calculated.  Then I concatenate log-mel feature with delta and accelerate to form a 3 x 128 x N dimension matrix where N depends on length of audio files. The feature was created by librosa's melspectrogram with parameters described in the link below.  

https://www.kaggle.com/daisukelab/fat2019_prep_mels1



## Model architecture

I used simple CNN described in the link below.   

https://www.kaggle.com/mhiro2/simple-2d-cnn-classifier-with-pytorch



## Training

The model was trained in several steps.  

Step1. Train the model with curated data and noisy data

- Augmentation: RandomCrop, mixup
- Loss: CrossEntropyLoss

Step2. Re-train the model  with only curated data

- Augmentation: RandomCrop, specAugment
- Loss: FocalLoss



## Winning solutions

リンク
