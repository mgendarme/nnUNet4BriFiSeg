# nnUNet4BriFiSeg

The best model for semantic and instance segmentation of nuclei in brightfield images developped using [BriFiSeg](https://github.com/mgendarme/BriFiSeg), namely U-Net SE ResNeXt 101 (FPN with SE ResNeXt 101 also provided) was ported here and made compatible with [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) functionalities. What we obtain is an architecture performing better than standard nnU-Net using U-Net with the classification model SE ResNeXt 101 trained on ImageNet compatible with nnU-Net for 2D semantic segmentation.

The data 2D images have to be preprocessed to be compatible with nnU-Net. The three channels of the RGB image need to be split in three modalities. [Example](https://github.com/mgendarme/nnUNet4BriFiSeg/tree/master/Example/Data) of data structure.

To preprocess (no preprocessing done on purpose, images are transformed to RGB images and processed to match ImageNet requirements as done in original publication of SE ResNeXt) and train nnU-Net commands were used as done [here](https://github.com/mgendarme/nnUNet4BriFiSeg/blob/master/Preprocess_and_train.sh).


