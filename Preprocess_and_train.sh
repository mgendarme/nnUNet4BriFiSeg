# Data preprocessing
# we do not want nnunet preprocessing, data should already have been preprocessed 
# with ImageNet preprocessing compatible with the encoder at use
# we did work only with 2D images
# replace everywhere in this file XXX with task ID of choice
nnUNet_plan_and_preprocess -t XXX -pl3d None -pl2d ExperimentPlanner2D_v21_RGB_nonorm

# for training with UNet
# be careful we modified the training procedure to be limited to 200 epoch
# example for fold 0
nnUNet_train 2d nnUNetTrainerV2_unet_v3_noDeepSupervision_sn_adam TaskXXX 0

# for training with FPN
# be careful we modified the training procedure to be limited to 200 epoch
# example for fold 0
nnUNet_train 2d nnUNetTrainerV2_fpn_noDeepSupervision_sn TaskXXX 0

# everything else has to be done with regular commands from nnunet