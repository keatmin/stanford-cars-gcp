# stanford-cars-gcp
Image classification using fastai v1 running on Google Cloud Platform VM NVIDIA Tesla P1. A better results are found with training with rectangular images after training it on square images and doubling the number of images in the dataset with (cropped and original images) sees an increase in accuracy of the model (94.27% TTA accuracy).  

This **readme** is arranged in such order: 
1. Results summary
2. Installation guide 
3. Test guide 
4. Training summary

## 1. Results Summary
All training is done using 1-cycle policy. 
Preprocessing methods and EDA is done in *0_preprocessing_and_EDA.ipynb*

Training time is calculated based on the time it takes for on the GCP VM NVIDIA Tesla P1 

Notebook | Details | Test Accuracy | Precision | Recall | Epochs | Training time
:---: | :---: | :---: | :---: | :---: | :---: | :---:
1_resnet50 | ResNet-50 Benchmark model  | 82.61% | x | x | 35e | 49 minutes
2_resnet152 | ResNet-152 Benchmark | 87.56% | x | x | 35e |  104 minutes 
3_resnet152_prog_sq-rect | Progressive resizing & square to rectangular images| **92.45%**(TTA) | 92.39% | 92.12% | 35e | 240 minutes
4_resnet152_prog_rect | Progressive resizing of only rectangular images | 91.69% | x | x | 35e | 207 minutes 
5_resnet152_prog_rect_cropped | Progressive resizing with cropped and normal images (16288 images)  | **94.27%**(TTA) | 93.83% | 93.57% | 35e | 477 minutes

## 2. Installation guide

fastai v1 requires at least pytorch v1.x and Python 3.6 and above. The library only works on Linux machines and it's in experimental stage on Windows machines.  

pip
```
pip install -r requirements.txt
```
Requirements


## 3. Testing Guide
Model can be downloaded via this link : [Google Drive link of model](https://drive.google.com/uc?export=download&confirm=ZfKU&id=1SiT2sHE6JDokx7m3qdlMTJJ0uAv5F73i). Place the file in the same directory as the test script `get_cars_predictions.py`

This test will be carried out via CPU based on the model from notebook 5 by default. It will generate a csv file of prediction class, probability and its filename in the current directory.
```python get_car_predictions.py 'holdout_testset_path' --csv_fname='csv_name_to_generate' ```

## 4. Training Summary
### Training
1. Initial hypothesis is training first with normal images followed by cropped image will have a higher accuracy, lesson learnt from this training is that during resizing, the image is zoomed in and cropped to the size that was chosen during resizing, losing features that make a car a car. 
2. Due to the low amount of data (8144 in training set and 196 classes, averaging 41 images per class for training and validation refer to EDA in the the first notebook *0_preprocessing_and_EDA.ipynb*), transfer learning was chosen on a pretrained ResNet-50 or a Resnet-152 model because training a network from scratch will not yield a very accurate result. 
3. `TTA()` is a technique that was used to test to increase the accuracy of the prediction by performing random transformation to test images based on augmentations and transformations done on the training set
4. Due to the low amount of training data for each class, I have decided to double the amount as well as data augmentation by using the cropped images based on bbox to increase the training dataset to 16288 images (1 cropped image for each normal image). Which worked well

### Data augmentation 
#### On training with rectangular images
Training with rectangular images showed to provide a faster convergence compared to training with a fixed square 224x224 images, _fastai_ library allows custom sizing and the pretrained ResNet-152 or ResNet-50 to train on rectangular images. Initial EDA showed that the median aspect ratio for the training dataset is 3:4. 

A typical transfer learning process involve training the last layer before unfreezing the whole architecture for training, which would be the typical approach of freezing the pretrained model and training it for 5 epochs before unfreezing the model

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
