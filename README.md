# stanford-cars-gcp
Stanford cars classifiers

requirements.txt 

`pip install fastai`

## Preprocessing 

## Training 
1. Initial hypothesis is training first with normal images followed by cropped image will have a higher accuracy, lesson learnt from this training is that during resizing, the image is zoomed in and cropped to the size that was chosen during resizing, losing features that make a car a car. 
2. 

## To implement 
 ~max_zoom = 1.05~ 


div factor 10
lrs = np.array([lr[1]/(div_lr**2), lr[1]/div_lr, lr[1]])
train with cropped images, tfms without zoom 
mixup
dropout
resize with a tuple to squish for cropped images
Crop 

TODO: 
Loading images, predicting results and returning probability

Resnext101
## Baseline with resnet50

training a network from scratch will not yield very accurate results
test set is transformed with get_transform
wd = 1e-3 is too low and may overfit

using rectangular aspect ratio
