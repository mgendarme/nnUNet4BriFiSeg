import os
import sys
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d, Sequential, UpsamplingBilinear2d
import torch.nn.functional

## FPN architecture and SE networks coming from:
## https://github.com/selimsef/xview2_solution
from nnunet.fpn_architecture.unet import Conv1x1, Conv3x3, ConvReLu3x3, encoder_params, ResneXt, Resnet, SeResneXt, SCSeResneXt
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.initialization import InitWeights_He


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, activation=nn.ReLU, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        # in 2048  out 1024
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, affine=True, momentum=0.1)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, affine=True, momentum=0.1)
        self.act2 = activation()   

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class DoubleConvBlockNnunet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, activation=nn.LeakyReLU, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        # in 2048  out 1024
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)
        self.in1 = nn.InstanceNorm2d(num_features=out_channels, eps=1e-5, affine=True)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)
        self.in2 = nn.InstanceNorm2d(num_features=out_channels, eps=1e-5, affine=True)
        self.act2 = activation()   

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.act2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Unet_SeResNeXt(SegmentationNetwork):
    def __init__(self,
                 seg_classes,
                 backbone_arch,
                 num_classes,
                 attention_mode='se',
                 weightInitializer=InitWeights_He(1e-2),
                 deep_supervision = False,
                 conv_op = nn.Conv2d,
                 drpout = 0.0):
        super(Unet_SeResNeXt, self).__init__()

        # necessary bits related to Generic_Unet and SegmentationNetwork
        self.do_ds = False # necessary to not implement deep_supervion
        self._deep_supervision = deep_supervision # necessary to not implement deep_supervion
        self.conv_op = conv_op # necessary for validate to determine if 2d or 3d net
        self.num_classes = num_classes # necessary to run validate

        ## get encoder
        if attention_mode == 'se':
            encoder = SeResneXt(seg_classes=seg_classes, num_channels=3, backbone_arch="seresnext101")
        elif attention_mode == 'scse':
            encoder = SCSeResneXt(seg_classes=seg_classes, num_channels=3, backbone_arch="seresnext101")

        # input conv
        self.input_conv = DoubleConvBlock(in_channels=3, out_channels=32)   # 32

        ## down path
        self.enc_conv0 = encoder.encoder_stages[0]
        self.enc_conv1 = encoder.encoder_stages[1]
        self.enc_conv2 = encoder.encoder_stages[2]
        self.enc_conv3 = encoder.encoder_stages[3]
        
        ## bottleneck
        self.enc_conv4 = encoder.encoder_stages[4]

        # nn.functional.interpolate instead of upsampling
        self.up_sample4 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample3 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample2 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample1 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample0 = Upsample(scale_factor=2, mode='bilinear')
        
        ## up path                                                                  # shapes: W or H
        self.dec_conv4 = DoubleConvBlock(in_channels=2048+1024, out_channels=512)   # 32
        self.dec_conv3 = DoubleConvBlock(in_channels=512+512, out_channels=256)     # 64
        self.dec_conv2 = DoubleConvBlock(in_channels=256+256, out_channels=128)     # 128
        self.dec_conv1 = DoubleConvBlock(in_channels=128+64, out_channels=64)       # 256
        self.dec_conv0 = DoubleConvBlock(in_channels=32+64, out_channels=32)        # 512
        self.final = Conv1x1(in_channels=32, out_channels=seg_classes)

        self.apply(weightInitializer)  

    def forward(self, x):               # shape: W or H
        x_input = self.input_conv(x)    # 512
        x0 = self.enc_conv0(x)          # 256
        x1 = self.enc_conv1(x0)         # 128
        x2 = self.enc_conv2(x1)         # 64
        x3 = self.enc_conv3(x2)         # 32
        
        x4 = self.enc_conv4(x3)         # 16

        up4 = self.up_sample4(x4)       # 32

        dec4 = torch.cat((up4, x3), dim = 1)
        dec4 = self.dec_conv4(dec4)
        up3 = self.up_sample3(dec4)     # 64

        dec3 = torch.cat((up3, x2), dim = 1)
        dec3 = self.dec_conv3(dec3)
        up2 = self.up_sample3(dec3)     # 128

        dec2 = torch.cat((up2, x1), dim = 1)
        dec2 = self.dec_conv2(dec2)
        up1 = self.up_sample3(dec2)     # 256

        dec1 = torch.cat((up1, x0), dim = 1)
        dec1 = self.dec_conv1(dec1)
        up0 = self.up_sample3(dec1)     # 512

        dec0  = torch.cat((up0, x_input), dim = 1)
        dec0 = self.dec_conv0(dec0)
        
        x = self.final(dec0)
        return x

# input = torch.randn(5, 3, 512, 512)
# unet_se = Unet_SeResNeXt(seg_classes=2, backbone_arch='seresnext101', num_classes=2, attention_mode='scse')
# unet_se(input).shape

class Unet_SeResNeXt_v2(SegmentationNetwork):
    def __init__(self,
                 seg_classes,
                 backbone_arch,
                 num_classes,
                 attention_mode='se',
                 weightInitializer=InitWeights_He(1e-2),
                 deep_supervision = False,
                 conv_op = nn.Conv2d,
                 drpout = 0.0):
        super(Unet_SeResNeXt_v2, self).__init__()

        # necessary bits related to Generic_Unet and SegmentationNetwork
        self.do_ds = False # necessary to not implement deep_supervion
        self._deep_supervision = deep_supervision # necessary to not implement deep_supervion
        self.conv_op = conv_op # necessary for validate to determine if 2d or 3d net
        self.num_classes = num_classes # necessary to run validate

        ## get encoder
        if attention_mode == 'se':
            encoder = SeResneXt(seg_classes=seg_classes, num_channels=3, backbone_arch="seresnext101")
        elif attention_mode == 'scse':
            encoder = SCSeResneXt(seg_classes=seg_classes, num_channels=3, backbone_arch="seresnext101")

        # input conv
        self.input_conv = DoubleConvBlock(in_channels=3, out_channels=3)   # 32

        ## down path
        self.enc_conv0 = encoder.encoder_stages[0]
        self.enc_conv1 = encoder.encoder_stages[1]
        self.enc_conv2 = encoder.encoder_stages[2]
        self.enc_conv3 = encoder.encoder_stages[3]
        
        ## bottleneck
        self.enc_conv4 = encoder.encoder_stages[4]

        # nn.functional.interpolate instead of upsampling
        self.up_sample4 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample3 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample2 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample1 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample0 = Upsample(scale_factor=2, mode='bilinear')
        
        ## up path                                                                  # shapes: W or H
        self.dec_conv4 = DoubleConvBlock(in_channels=2048+1024, out_channels=512)   # 32
        self.dec_conv3 = DoubleConvBlock(in_channels=512+512, out_channels=256)     # 64
        self.dec_conv2 = DoubleConvBlock(in_channels=256+256, out_channels=128)     # 128
        self.dec_conv1 = DoubleConvBlock(in_channels=128+64, out_channels=64)       # 256
        self.dec_conv0 = DoubleConvBlock(in_channels=3+64, out_channels=32)         # 512
        self.final = Conv1x1(in_channels=32, out_channels=seg_classes)

        self.apply(weightInitializer)  

    def forward(self, x):               # shape: W or H
        x_input = self.input_conv(x)    # 512
        x0 = self.enc_conv0(x_input)    # 256
        x1 = self.enc_conv1(x0)         # 128
        x2 = self.enc_conv2(x1)         # 64
        x3 = self.enc_conv3(x2)         # 32
        
        x4 = self.enc_conv4(x3)         # 16

        up4 = self.up_sample4(x4)       # 32

        dec4 = torch.cat((up4, x3), dim = 1)
        dec4 = self.dec_conv4(dec4)
        up3 = self.up_sample3(dec4)     # 64

        dec3 = torch.cat((up3, x2), dim = 1)
        dec3 = self.dec_conv3(dec3)
        up2 = self.up_sample3(dec3)     # 128

        dec2 = torch.cat((up2, x1), dim = 1)
        dec2 = self.dec_conv2(dec2)
        up1 = self.up_sample3(dec2)     # 256

        dec1 = torch.cat((up1, x0), dim = 1)
        dec1 = self.dec_conv1(dec1)
        up0 = self.up_sample3(dec1)     # 512

        dec0  = torch.cat((up0, x_input), dim = 1) 
        dec0 = self.dec_conv0(dec0)
        
        x = self.final(dec0)
        return x


class Unet_SeResNeXt_v3(SegmentationNetwork):
    def __init__(self,
                 seg_classes,
                 backbone_arch,
                 num_classes,
                 attention_mode='se',
                 weightInitializer=InitWeights_He(1e-2),
                 deep_supervision = False,
                 conv_op = nn.Conv2d,
                 drpout = 0.0):
        super(Unet_SeResNeXt_v3, self).__init__()

        # necessary bits related to Generic_Unet and SegmentationNetwork
        self.do_ds = False # necessary to not implement deep_supervion
        self._deep_supervision = deep_supervision # necessary to not implement deep_supervion
        self.conv_op = conv_op # necessary for validate to determine if 2d or 3d net
        self.num_classes = num_classes # necessary to run validate

        ## get encoder
        if attention_mode == 'se':
            encoder = SeResneXt(seg_classes=seg_classes, num_channels=3, backbone_arch="seresnext101")
        elif attention_mode == 'scse':
            encoder = SCSeResneXt(seg_classes=seg_classes, num_channels=3, backbone_arch="seresnext101")

        # input conv
        self.input_conv = DoubleConvBlockNnunet(in_channels=3, out_channels=3)   # 32

        ## down path
        self.enc_conv0 = encoder.encoder_stages[0]
        self.enc_conv1 = encoder.encoder_stages[1]
        self.enc_conv2 = encoder.encoder_stages[2]
        self.enc_conv3 = encoder.encoder_stages[3]
        
        ## bottleneck
        self.enc_conv4 = encoder.encoder_stages[4]

        # nn.functional.interpolate instead of upsampling
        self.up_sample4 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample3 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample2 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample1 = Upsample(scale_factor=2, mode='bilinear')
        self.up_sample0 = Upsample(scale_factor=2, mode='bilinear')
        
        ## up path                                                                  # shapes: W or H
        self.dec_conv4 = DoubleConvBlockNnunet(in_channels=2048+1024, out_channels=512)   # 32
        self.dec_conv3 = DoubleConvBlockNnunet(in_channels=512+512, out_channels=256)     # 64
        self.dec_conv2 = DoubleConvBlockNnunet(in_channels=256+256, out_channels=128)     # 128
        self.dec_conv1 = DoubleConvBlockNnunet(in_channels=128+64, out_channels=64)       # 256
        self.dec_conv0 = DoubleConvBlockNnunet(in_channels=3+64, out_channels=32)         # 512
        self.final = Conv1x1(in_channels=32, out_channels=seg_classes)

        self.apply(weightInitializer)  

    def forward(self, x):               # shape: W or H
        x_input = self.input_conv(x)    # 512
        x0 = self.enc_conv0(x_input)    # 256
        x1 = self.enc_conv1(x0)         # 128
        x2 = self.enc_conv2(x1)         # 64
        x3 = self.enc_conv3(x2)         # 32
        
        x4 = self.enc_conv4(x3)         # 16

        up4 = self.up_sample4(x4)       # 32

        dec4 = torch.cat((up4, x3), dim = 1)
        dec4 = self.dec_conv4(dec4)
        up3 = self.up_sample3(dec4)     # 64

        dec3 = torch.cat((up3, x2), dim = 1)
        dec3 = self.dec_conv3(dec3)
        up2 = self.up_sample3(dec3)     # 128

        dec2 = torch.cat((up2, x1), dim = 1)
        dec2 = self.dec_conv2(dec2)
        up1 = self.up_sample3(dec2)     # 256

        dec1 = torch.cat((up1, x0), dim = 1)
        dec1 = self.dec_conv1(dec1)
        up0 = self.up_sample3(dec1)     # 512

        dec0  = torch.cat((up0, x_input), dim = 1) 
        dec0 = self.dec_conv0(dec0)
        
        x = self.final(dec0)
        return x
