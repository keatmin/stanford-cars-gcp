# stanford-cars-gcp
Image classification using fastai v1 run on GCP VM P1. 

This readme is arranged in such order: 
1. Results summary
2. Installation guide 
3. Test guide 
4. Preprocessing and training details

# Results Summary
All training is done using 1-cycle policy. 
Preprocessing methods and EDA is done in *0_preprocessing_and_EDA.ipynb*

Notebook | Details | Accuracy | Precision | Recall | Epochs | Training time
*** | *** | *** | *** | *** | *** | 
1_resnet50 | ResNet-50 Benchmark model  | 82.61% | x | x | 35e | 49 minutes
2_resnet152 | ResNet-152 Benchmark | 87.56% | x | x | 35e |  104 minutes 
3_resnet152 | ResNet-152 progressive resizing & square-rectangular image training | 92.45% | x | x | 35e | 





requirements.txt 

CPU build
Generally, pytorch GPU build should work fine on machines that don’t have a CUDA-capable GPU, and will just use the CPU. However, you can install CPU-only versions of Pytorch if needed with fastai.

pip
```
pip install http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
pip install fastai 
```

Just make sure to pick the correct torch wheel url, according to the needed platform, python and CUDA version, which you will find here.

conda

The conda way is more involved. Since we have only a single fastai package that relies on the default pytorch package working with and without GPU environment, if you want to install something custom you will have to manually tweak the dependencies. This is explained in detail here. So follow the instructions there, but replace pytorch with pytorch-cpu, and torchvision with torchvision-cpu.

Also, please note, that if you have an old GPU and pytorch fails because it can’t support it, you can still use the normal (GPU) pytorch build, by setting the env var CUDA_VISIBLE_DEVICES="", in which case pytorch will not try to check if you even have a GPU.

## Preprocessing 

## Training 
1. Initial hypothesis is training first with normal images followed by cropped image will have a higher accuracy, lesson learnt from this training is that during resizing, the image is zoomed in and cropped to the size that was chosen during resizing, losing features that make a car a car. 
2. Due to the low amount of 
3. What is TTA()


TODO: 
Loading images, predicting results and returning probability

## Baseline with resnet50

training a network from scratch will not yield very accurate results
test set is transformed with get_transform
wd = 1e-3 is too low and may overfit

using rectangular aspect ratio


