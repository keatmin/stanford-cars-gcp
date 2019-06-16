# stanford-cars-gcp
Image classification using fastai v1 running on Google Cloud Platform VM NVIDIA Tesla P1. 

- A better result is found with training with rectangular images (4:3 ratio) after training it on square images, cameras in the market **all** take photos in a **4:3** or **16:9** ratio by default unless the user voluntarily chooses a 1:1 ratio(a potential psychopath?). Hence this makes model trained with rectangular images appropriate to predict images taken from smartphone cameras as well. 
- Regularizer such as [mixup](https://arxiv.org/abs/1710.09412) was helpful in increasing the accuracy but it requires more epochs to achieve the same result, doubling the time to achieve the same result by just resizing to rectangular images. A combination of rectangular images and mixup will improve the accuracy of the model more but the opportunity cost is higher than training with the next method
- Doubling the number of images in the dataset with cropped and original images sees an increase in accuracy of the model (94.27% TTA accuracy). Refer to [Training summary](https://github.com/keatmin/stanford-cars-gcp#4-training-summary) for more information regarding this method. 

## Contents: 
1. [Results summary](https://github.com/keatmin/stanford-cars-gcp#1-results-summary)
2. [Installation guide](https://github.com/keatmin/stanford-cars-gcp#2-installation-guide)
3. [Test guide](https://github.com/keatmin/stanford-cars-gcp#3-testing-guide)
4. [Training summary](https://github.com/keatmin/stanford-cars-gcp#4-training-summary)

## 1. Results Summary
All training is done using [Leslie Smith's 1-cycle policy](https://arxiv.org/pdf/1803.09820).
Preprocessing methods and EDA is done in *0_preprocessing_and_EDA.ipynb*

Training time is calculated based on the time it takes for the GCP VM NVIDIA Tesla P1 to train the whole model.

Notebook | Details | Test Accuracy | Precision | Recall | Epochs | Training time
:---: | :---: | :---: | :---: | :---: | :---: | :---:
1_resnet50 | ResNet-50 Benchmark model  | 82.61% | x | x | 35e | 49 minutes
2_resnet152 | ResNet-152 Benchmark | 87.56% | x | x | 35e |  104 minutes 
3_resnet152_prog_sq-rect | Progressive resizing & square to rectangular images| **92.45%**(TTA) | 92.39% | 92.12% | 35e | 240 minutes
4_resnet152_prog_rect | Progressive resizing of only rectangular images | 91.69% | x | x | 35e | 207 minutes 
5_resnet152_prog_rect_cropped | Progressive resizing with cropped and normal images (16288 images)  | **94.27%**(TTA) | 93.83% | 93.57% | 35e | 477 minutes

## 2. Installation guide

fastai v1 requires at least `pytorch v1.x` and `Python 3.6` and above. The library only works on Linux machines and it's in experimental stage on Windows machines but **not** on MacOS

pip
```
pip install -r requirements.txt
```


## 3. Testing Guide
Model can be downloaded via this link : [Google Drive link of model from notebook 5](https://drive.google.com/uc?export=download&confirm=uVfv&id=1ZY9yt5Gtkvoy4HEtEqFVjZzopGLaMOPq). <br>

[Model from notebook 3](https://drive.google.com/uc?export=download&confirm=Z9RS&id=1Z_p_KaVUnsoBUkAgqHAFSXabfeN8Sgz0)<br>
[Model from notebook 4](https://drive.google.com/uc?export=download&confirm=Qdde&id=1HOMDiBiteQGX5l3Qi_bH6CysbqlyRX7h)

Place the file in the same directory as the test script `get_cars_predictions.py`

This test will be carried out via CPU based on the [model](https://drive.google.com/uc?export=download&confirm=uVfv&id=1ZY9yt5Gtkvoy4HEtEqFVjZzopGLaMOPq) from notebook 5 by default. It will generate a csv file of prediction class, probability and its filename in the current directory.

```
python get_cars_prediction.py 'holdout_testset_path'
```

`holdout_testset_path` being the folder consisting test images relative to the script. 

or 

```
python get_cars_prediction.py 'holdout_testset_path' --csv_fname='csv_name_to_generate.csv' --model='model_name.pkl' 
```

Several `model` can be obtained, but pickle file from notebook 5 should perform the best. 

## 4. Training Summary
### Training 

- Initial hypothesis is training first with normal images followed by cropped image will have a higher accuracy, lesson learnt from this training is that during resizing, the image is zoomed in and cropped to the size that was chosen during resizing, losing features that make a car a car. Hence training it at 299x299 first before resizing to 299x400 images work extremely well. 

- Due to the low amount of data (8144 in training set and 196 classes, averaging 41 images per class for training and validation refer to EDA in the the first notebook *0_preprocessing_and_EDA.ipynb*), transfer learning was chosen on a pretrained ResNet-50 or a Resnet-152 model because training a network from scratch will not yield a very accurate result. 

- `TTA()` is a technique that was used to test to increase the accuracy of the prediction by performing random transformation to test images based on augmentations and transformations done on the training set.

- mixup was a 

- Due to the low amount of training data for each class, I have decided to double the amount as well as data augmentation by using the cropped images based on bbox to increase the training dataset to 16288 images (1 cropped image for each normal image). Which worked well, the only downside is it takes 447 minutes(more than 7 hours), the expense of training time more than **doubled** and cropped images of the same car appearing in the validation set in order to maximise the amount of data, making validation score looks inflated and unreliable when training. **A mistake learnt** that is avoidable by using indexes or column in a dataframe to split rather than a random percentage split but due to the time constraint and time it takes to train, I was unable to alter the split (I am a human, afterall).Due to that, this method somehow **defies the traditional practice** of having train-validation-test split by maximising the number of available data (inspired by competitions in Kaggle and Agile development) by only having the available test set as the only reliable benchmark. 


### Data augmentation 
#### On training with rectangular images
Training with rectangular images showed to provide a faster convergence compared to training with a fixed square 224x224 images, _fastai_ library allows custom sizing and the pretrained ResNet-152 or ResNet-50 to train on rectangular images. Initial EDA showed that the median aspect ratio for the training dataset is 3:4. 

A typical transfer learning process involve training the last layer before unfreezing the whole architecture for training, which would be the typical approach of freezing the pretrained model and training it for 5 epochs before unfreezing the model

#### Training with cropped images
I decided to train normal images and cropped images together due to the fact that there are too little data available for each class to be trained. That way I doubled the number of images of each class, helping the model to recognise the noise from the actual object to classify. Together with data augmentation, the model will be able to obtain more dataset to increase it's accuracy. This method helped to increase the accuracy of my previous best training by **1.7%** 

#### Data Augmentation 
Perhaps the most important step that is essential in providing a better accuracy. Together with **progressive resizing** and **rectangular images training**, I have also used the default fastai images transformation, performed by calling `get_transforms()` that randomly augments the images according to their parameters: <br>

`do_flip` : perform flipping with probability of 0.5. I chose the default value **True** for this parameter <br>
`flip_ver` : if True flip images vertically instead, I chose **False** as it doesn't help to have a vertically flipped images<br>
`max_rotate`: I chose up to **10.0** degree rotation at random, <br>
`max_zoom`: default value of up to **1.1** zoom<br>
`max_lighting`: default value of up to **0.2** of increase/decrease in brightness or contrast<br>
`max_warp`:  default value of **0.2**<br>
`p_affine` : Probability of randomly applying affine, default value **0.75**<br>
`p_lighting` : probability of applying lighting transform, default value **0.75** <br>

## Credits
Codes to extract the labels from the .mat files: [Devon Yates' code on Kaggle](https://www.kaggle.com/criticalmassacre/inaccurate-labels-in-stanford-cars-data-set)
