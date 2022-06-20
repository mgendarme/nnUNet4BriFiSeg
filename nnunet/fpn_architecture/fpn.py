import os
import sys
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d, Sequential, UpsamplingBilinear2d
import torch.nn.functional
from nnunet.network_architecture.initialization import InitWeights_He


from nnunet.fpn_architecture.unet import Conv1x1, Conv3x3, ConvReLu3x3, encoder_params, ResneXt, Resnet, DPNUnet, \
    DensenetUnet, SCSeResneXt, SeResneXt

from nnunet.network_architecture.neural_network import SegmentationNetwork

class FPN(nn.Module):
    def __init__(self, inner_filters, filters):
        super().__init__()
        self.lateral4 = Conv1x1(filters[-1], 256)
        self.lateral3 = Conv1x1(filters[-2], 256)
        self.lateral2 = Conv1x1(filters[-3], 256)
        self.lateral1 = Conv1x1(filters[-4], 256)

        self.smooth5 = Conv3x3(256, inner_filters)
        self.smooth4 = Conv3x3(256, inner_filters)
        self.smooth3 = Conv3x3(256, inner_filters)
        self.smooth2 = Conv3x3(256, inner_filters)

    def forward(self, encoder_results: list):
        x = encoder_results[0]
        lateral4 = self.lateral4(x)
        lateral3 = self.lateral3(encoder_results[1])
        lateral2 = self.lateral2(encoder_results[2])
        lateral1 = self.lateral1(encoder_results[3])

        m5 = lateral4
        m4 = lateral3 + F.upsample(m5, scale_factor=2, mode="nearest")
        m3 = lateral2 + F.upsample(m4, scale_factor=2, mode="nearest")
        m2 = lateral1 + F.upsample(m3, scale_factor=2, mode="nearest")

        p5 = self.smooth5(m5)
        p4 = self.smooth4(m4)
        p3 = self.smooth3(m3)
        p2 = self.smooth2(m2)

        return p2, p3, p4, p5


class FPNSegmentation(nn.Module):

    def __init__(self, inner_filters, filters, forward_pyramid_features=False, upsampling_mode="nearest"):
        super().__init__()
        self.fpn = FPN(inner_filters, filters)
        self.forward_pyramid_features = forward_pyramid_features
        seg_filters = inner_filters // 2
        output_filters = seg_filters
        ##################################################
        for i in range(2, 6):
        ###### include module 1 too? #####################
            # print(i)
            self.add_module("level{}".format(i),
                            nn.Sequential(ConvReLu3x3(inner_filters, seg_filters),
                                          ConvReLu3x3(seg_filters, seg_filters)))
        self.upsampling = upsampling_mode
        self.aggregator = Sequential(
            Conv3x3(seg_filters * 4, output_filters),
            BatchNorm2d(output_filters),
            nn.ReLU()
        )

    def forward(self, encoder_results: list):
        pyramid_features = self.fpn(encoder_results)
        outputs = [
            self.level2(pyramid_features[0]),
            F.upsample(self.level3(pyramid_features[1]), scale_factor=2, mode=self.upsampling),
            F.upsample(self.level4(pyramid_features[2]), scale_factor=4, mode=self.upsampling),
            F.upsample(self.level5(pyramid_features[3]), scale_factor=8, mode=self.upsampling),
        ]
        x = torch.cat(outputs, dim=1)
        x = self.aggregator(x)
        return x

# class Decoder_block_no_bn(nn.Module):
#     def __init__(self, filters, skip):
#         super().__init__(filters, skip)
#         self.up = UpsamplingBilinear2d(scale_factor=2)
#         self.conv = Conv3x3(256, 128)

#     def forward(self, x):
#         x = self.up(x)
#         x = self.conv(x)
#         x = torch.cat([x, skip])
#         x = self.conv(x)
#         return x

def _initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data = nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class ResneXtFPN(ResneXt):
    def __init__(self, seg_classes, backbone_arch):
        super().__init__(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=256, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.final = Conv1x1(in_channels=128, out_channels=seg_classes)
        self.dropout = nn.Dropout2d(p=0.15)
        _initialize_weights(self.fpn)
        _initialize_weights(self.final)


    def forward(self, x):
        enc_results = []
        ###### problematic piece ################################################################
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
                # enc_results.append(x.clone())
        #########################################################################################
        seg = self.fpn(list(reversed(enc_results)))
        seg = self.up(seg)
        seg = self.dropout(seg)
        x = self.final(seg)
        return x

class SEResNeXtFPN(SeResneXt):
    def __init__(self, seg_classes, backbone_arch, deep_supervision = False):
        super().__init__(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=256, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.final = Conv1x1(in_channels=128, out_channels=seg_classes)
        self.dropout = nn.Dropout2d(p=0.15)
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        _initialize_weights(self.fpn)
        _initialize_weights(self.final)

    def forward(self, x):
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
            # enc_results.append(x.clone())
        seg = self.fpn(list(reversed(enc_results)))
        seg = self.up(seg)
        seg = self.dropout(seg)
        x = self.final(seg)
        return x

class SEResNeXtFPNdsb(SCSeResneXt):
    def __init__(self, seg_classes, backbone_arch):
        super().__init__(seg_classes, backbone_arch, deep_supervision = False)
        self.fpn = FPNSegmentation(inner_filters=256, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.conv_skip = Conv3x3(in_channels=256, out_channels=128)
        self.conv_post_skip = Conv3x3(in_channels=256, out_channels=128)
        self.conv = Conv3x3(in_channels=128, out_channels=128)
        self.conv_post_1 = Conv3x3(in_channels=128, out_channels=128)
        self.conv_post_2 = Conv3x3(in_channels=128, out_channels=64)
        self.final = Conv1x1(in_channels=64, out_channels=seg_classes)
        self.dropout = nn.Dropout2d(p=0.15)
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        _initialize_weights(self.fpn)
        _initialize_weights(self.final)        

    def forward(self, x):
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
            # enc_results.append(x.clone())
        seg = self.fpn(list(reversed(enc_results)))                                 # 128
        # seg = self.up(seg)                                    # added from dsb    # 128
        seg = self.conv(seg)                                    # added from dsb    # 128
        skip = self.conv_skip(enc_results[0])                   # added from dsb    # 128 to reduce skip from 256 to 128
        seg = torch.cat([seg, skip], dim = 1)                   # added from dsb    # 128
        seg = self.conv_post_skip(seg)                               # added from dsb    # 128
        seg = self.up(seg)
        seg = self.dropout(seg)
        seg = self.conv_post_1(seg)                             # added from dsb    # 64
        seg = self.conv_post_2(seg)                             # added from dsb    # 64
        x = self.final(seg)                                                         # #of classe
        return x

class SCSEResNeXtFPNdsb2(SCSeResneXt):
    def __init__(self, seg_classes, backbone_arch,
                 deep_supervision = False,
                 weightInitializer=InitWeights_He(1e-2)):
        super().__init__(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=256, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.dec = Sequential(
            Conv3x3(in_channels=128, out_channels=128),

        )
        self.conv_skip = Conv3x3(in_channels=256, out_channels=128)
        self.conv = Conv3x3(in_channels=128, out_channels=128)
        self.dropout = nn.Dropout2d(p=0.15)
        self.conv_post_1 = Conv3x3(in_channels=128, out_channels=128)
        self.conv_post_2 = Conv3x3(in_channels=128, out_channels=64)
        self.final = Conv1x1(in_channels=64, out_channels=seg_classes)
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        # self.weightInitializer = weightInitializer
        _initialize_weights(self.fpn)
        _initialize_weights(self.final)        

    def forward(self, x):
        enc_results = []
        x = torch.squeeze(x)
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
            # enc_results.append(x.clone())
        seg = self.fpn(list(reversed(enc_results)))                                 # 128
        # seg = self.up(seg)                                    # added from dsb    # 128
        seg = self.conv(seg)                                    # added from dsb    # 128
        skip = self.conv_skip(enc_results[0])                   # added from dsb    # 128 to reduce skip from 256 to 128
        seg = torch.cat([seg, skip], dim = 1)                   # added from dsb    # 128
        seg = self.conv_skip(seg)                               # added from dsb    # 128
        seg = self.up(seg)
        seg = self.dropout(seg)
        seg = self.conv_post_1(seg)                             # added from dsb    # 64
        seg = self.conv_post_2(seg)                             # added from dsb    # 64
        x = self.final(seg)                                                         # #of classe
        return x

# class SCSEResNeXtFPNdsb3(SCSeResneXt):
#     def __init__(self, seg_classes, backbone_arch,
#                  deep_supervision = False,
#                  weightInitializer=InitWeights_He(1e-2)):
#         super(SCSEResNeXtFPNdsb3, self).__init__()
#         self.fpn = FPNSegmentation(inner_filters=256, filters=encoder_params[backbone_arch]["filters"])
#         self.up = UpsamplingBilinear2d(scale_factor=4)
#         self.dec = Sequential(
#             Conv3x3(in_channels=128, out_channels=128),

#         )
#         self.conv_skip = Conv3x3(in_channels=256, out_channels=128)
#         self.conv = Conv3x3(in_channels=128, out_channels=128)
#         self.conv_post_1 = Conv3x3(in_channels=128, out_channels=128)
#         self.conv_post_2 = Conv3x3(in_channels=128, out_channels=64)
#         self.final = Conv1x1(in_channels=64, out_channels=seg_classes)
#         self.dropout = nn.Dropout2d(p=0.15)
#         self._deep_supervision = deep_supervision
#         self.do_ds = deep_supervision
#         # self.weightInitializer = weightInitializer
#         _initialize_weights(self.fpn)
#         _initialize_weights(self.final)        

#     def forward(self, x):
#         enc_results = []
#         x = torch.squeeze(x)
#         for i, stage in enumerate(self.encoder_stages):
#             x = stage(x)
#             if i > 0:
#                 enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
#             # enc_results.append(x.clone())
#         seg = self.fpn(list(reversed(enc_results)))                                 # 128
#         # seg = self.up(seg)                                    # added from dsb    # 128
#         seg = self.conv(seg)                                    # added from dsb    # 128
#         skip = self.conv_skip(enc_results[0])                   # added from dsb    # 128 to reduce skip from 256 to 128
#         seg = torch.cat([seg, skip], dim = 1)                   # added from dsb    # 128
#         seg = self.conv_skip(seg)                               # added from dsb    # 128
#         seg = self.up(seg)
#         seg = self.dropout(seg)
#         seg = self.conv_post_1(seg)                             # added from dsb    # 64
#         seg = self.conv_post_2(seg)                             # added from dsb    # 64
#         x = self.final(seg)                                                         # #of classe
#         return x

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class SEResNeXtFPNdsb2(SeResneXt):
    def __init__(self, seg_classes, backbone_arch,
                 weightInitializer=InitWeights_He(1e-2),
                 deep_supervision = False):
        super(SEResNeXtFPNdsb2, self).__init__(seg_classes, backbone_arch)

        self.do_ds = False

        self.fpn = FPNSegmentation(inner_filters=256, filters=encoder_params[backbone_arch]["filters"])
        # self.up = UpsamplingBilinear2d(scale_factor=4)
        # nn.functional.interpolate instead of upsampling
        self.up = Upsample(scale_factor=4, mode='bilinear')
        self.dec = Conv3x3(in_channels=128, out_channels=128)
        self.conv_skip = Conv3x3(in_channels=256, out_channels=128)
        self.conv = Conv3x3(in_channels=128, out_channels=128)
        self.conv_post_1 = Conv3x3(in_channels=128, out_channels=128)
        self.conv_post_2 = Conv3x3(in_channels=128, out_channels=64)
        self.final = Conv1x1(in_channels=64, out_channels=seg_classes)
        self.dropout = nn.Dropout2d(p=0.15)
        self._deep_supervision = deep_supervision
        # self.predict_3D = SegmentationNetwork.predict_3D()
        
        self.apply(weightInitializer)
        # _initialize_weights(self.fpn)
        # _initialize_weights(self.final)        

    def forward(self, x):
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
            # enc_results.append(x.clone())
        seg = self.fpn(list(reversed(enc_results)))                                 # 128
        # seg = self.up(seg)                                    # added from dsb    # 128
        seg = self.conv(seg)                                    # added from dsb    # 128
        skip = self.conv_skip(enc_results[0])                   # added from dsb    # 128 to reduce skip from 256 to 128
        seg = torch.cat([seg, skip], dim = 1)                   # added from dsb    # 128
        seg = self.conv_skip(seg)                               # added from dsb    # 128
        seg = self.up(seg)
        seg = self.dropout(seg)
        seg = self.conv_post_1(seg)                             # added from dsb    # 64
        seg = self.conv_post_2(seg)                             # added from dsb    # 64
        x = self.final(seg)                                                         # #of classe
        return x

class SEResNeXtFPNdsb3(SegmentationNetwork):
    def __init__(self, seg_classes, backbone_arch, num_classes,
                 weightInitializer=InitWeights_He(1e-2),
                 deep_supervision = False,
                 conv_op = nn.Conv2d,
                 drpout = 0.0):
        super(SEResNeXtFPNdsb3, self).__init__()

        # necessary bits related to Generi_Unet and SegmentationNetwork
        self.do_ds = False # necessary to not implement deep_supervion
        self._deep_supervision = deep_supervision # necessary to not implement deep_supervion
        self.conv_op = conv_op # necessary for validate to determine if 2d or 3d net
        self.num_classes = num_classes # necessary to run validate
                
        self.enc = SeResneXt(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=256, filters=encoder_params[backbone_arch]["filters"])
        # nn.functional.interpolate instead of upsampling
        self.up = Upsample(scale_factor=4, mode='bilinear')
        self.dec = Conv3x3(in_channels=128, out_channels=128)
        self.conv_skip = Conv3x3(in_channels=256, out_channels=128)
        self.conv = Conv3x3(in_channels=128, out_channels=128)
        self.dropout = nn.Dropout2d(p=drpout)
        self.conv_post_1 = Conv3x3(in_channels=128, out_channels=128)
        self.conv_post_2 = Conv3x3(in_channels=128, out_channels=64)
        self.final = Conv1x1(in_channels=64, out_channels=seg_classes)

        self.apply(weightInitializer)
        # _initialize_weights(self.fpn)
        # _initialize_weights(self.final)        

    def forward(self, x):
        # print('Model = SEResNeXtFPNdsb3')
        encoder = self.enc
        self.encoder_stages = encoder.encoder_stages
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
            # enc_results.append(x.clone())
        seg = self.fpn(list(reversed(enc_results)))                                 # 128
        # seg = self.up(seg)                                    # added from dsb    # 128
        seg = self.conv(seg)                                    # added from dsb    # 128
        skip = self.conv_skip(enc_results[0])                   # added from dsb    # 128 to reduce skip from 256 to 128
        seg = torch.cat([seg, skip], dim = 1)                   # added from dsb    # 128
        seg = self.conv_skip(seg)                               # added from dsb    # 128
        seg = self.up(seg)
        seg = self.dropout(seg)
        seg = self.conv_post_1(seg)                             # added from dsb    # 64
        seg = self.conv_post_2(seg)                             # added from dsb    # 64
        x = self.final(seg)                                                         # #of classe
        return x

class SCSEResNeXtFPNdsb3(SegmentationNetwork):
    def __init__(self, seg_classes, backbone_arch, num_classes,
                 weightInitializer=InitWeights_He(1e-2),
                 deep_supervision = False,
                 conv_op = nn.Conv2d):
        super(SCSEResNeXtFPNdsb3, self).__init__()

        # necessary bits related to Generi_Unet and SegmentationNetwork
        self.do_ds = False # necessary to not implement deep_supervion
        self._deep_supervision = deep_supervision # necessary to not implement deep_supervion
        self.conv_op = conv_op # necessary for validate to determine if 2d or 3d net
        self.num_classes = num_classes # necessary to run validate
                
        self.enc = SCSeResneXt(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=256, filters=encoder_params[backbone_arch]["filters"])
        # nn.functional.interpolate instead of upsampling
        self.up = Upsample(scale_factor=4, mode='bilinear')
        self.dec = Conv3x3(in_channels=128, out_channels=128)
        self.conv_skip = Conv3x3(in_channels=256, out_channels=128)
        self.conv = Conv3x3(in_channels=128, out_channels=128)
        self.conv_post_1 = Conv3x3(in_channels=128, out_channels=128)
        self.conv_post_2 = Conv3x3(in_channels=128, out_channels=64)
        self.final = Conv1x1(in_channels=64, out_channels=seg_classes)
        self.dropout = nn.Dropout2d(p=0.0)

        self.apply(weightInitializer)
        # _initialize_weights(self.fpn)
        # _initialize_weights(self.final)        

    def forward(self, x):
        # print('Model = SCSEResNeXtFPNdsb3')
        encoder = self.enc
        self.encoder_stages = encoder.encoder_stages
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
            # enc_results.append(x.clone())
        seg = self.fpn(list(reversed(enc_results)))                                 # 128
        # seg = self.up(seg)                                    # added from dsb    # 128
        seg = self.conv(seg)                                    # added from dsb    # 128
        skip = self.conv_skip(enc_results[0])                   # added from dsb    # 128 to reduce skip from 256 to 128
        seg = torch.cat([seg, skip], dim = 1)                   # added from dsb    # 128
        seg = self.conv_skip(seg)                               # added from dsb    # 128
        seg = self.up(seg)
        seg = self.dropout(seg)
        seg = self.conv_post_1(seg)                             # added from dsb    # 64
        seg = self.conv_post_2(seg)                             # added from dsb    # 64
        x = self.final(seg)                                                         # #of classe
        return x

# testenc = SeResneXt(3, 'seresnext101')
# testenc.encoder_stages

# class NetWrapper(SegmentationNetwork):
#     def __init__(self, seg_classes, backbone_arch):
#         super(NetWrapper, self).__init__(seg_classes, backbone_arch)

#         self.net = SEResNeXtFPNdsb2(seg_classes, backbone_arch)

#     # def forward(self, x):
#     #     # x = self.net(x)
#     #     res = super(NetWrapper, self).forward(x)
#     #     return res

# NetWrapper(seg_classes=3, backbone_arch='seresnext101')

class ResnetFPN(Resnet):
    def __init__(self, seg_classes, backbone_arch):
        super().__init__(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=128, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.final = Conv1x1(in_channels=128, out_channels=seg_classes)
        self.dropout = nn.Dropout2d(p=0.5)
        _initialize_weights(self.fpn)
        _initialize_weights(self.final)

    def forward(self, x):
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        seg = self.fpn(list(reversed(enc_results)))
        seg = self.up(seg)
        seg = self.dropout(seg)
        x = self.final(seg)
        return x


class DPNFPN(DPNUnet):
    def __init__(self, seg_classes, backbone_arch):
        super().__init__(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=128, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.final = Conv1x1(in_channels=128, out_channels=seg_classes)

    def forward(self, x):
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        seg = self.fpn(list(reversed(enc_results)))
        seg = self.up(seg)
        x = self.final(seg)
        return x

class DensenetFPN(DensenetUnet):
    def __init__(self, seg_classes, backbone_arch):
        super().__init__(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=128, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.final = Conv1x1(in_channels=128, out_channels=seg_classes)

    def forward(self, x):
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        seg = self.fpn(list(reversed(enc_results)))
        seg = self.up(seg)
        x = self.final(seg)
        return x


setattr(sys.modules[__name__], 'seresnext_fpn', partial(SEResNeXtFPN))
setattr(sys.modules[__name__], 'seresnext_fpn_dsb', partial(SEResNeXtFPNdsb))
setattr(sys.modules[__name__], 'resnet_fpn', partial(ResnetFPN))

__all__ = ['seresnext_fpn',
           'seresnext_fpn_dsb', 
           'resnet_fpn',
           ]

#####################################################################################################
# conf
# encoder_params
# class SEResNeXtFPN(SCSeResneXt):
#     def __init__(self, seg_classes, backbone_arch):
#         super().__init__(seg_classes, backbone_arch)
        
#         fpn = FPNSegmentation(inner_filters=256, filters=[64, 256, 512, 1024, 2048])
#         up = UpsamplingBilinear2d(scale_factor=4)
#         final = Conv1x1(in_channels=128, out_channels=1)
#         dropout = nn.Dropout2d(p=0.15)
#         _initialize_weights(fpn)
#         _initialize_weights(final)


# img = np.zeros((512, 512, 3), dtype='int')

#     def forward(self, x):
#         enc_results = []
#         x = fpn
#         stage0 = testmod.encoder_stages[0]
#         stage1 = testmod.encoder_stages[1]
#         stage2 = testmod.encoder_stages[2]
#         stage3 = testmod.encoder_stages[3]
#         stage4 = testmod.encoder_stages[4]
#         x = stage0(x)
#         for i, stage in enumerate(testmod.encoder_stages):
#             # print('stage ' + str(i) + ' we are here --------------------------------------------')
#             # print(stage)
#             x = stage(x)
#             if i > 0:
#                 enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
#         seg = self.fpn(list(reversed(enc_results)))
#         seg = self.up(seg)
#         seg = self.dropout(seg)
#         x = self.final(seg)
#         return x

# print(testmod.encoder_stages)

# input = torch.randn(5, 3, 512, 512)
# fpn = FPNSegmentation(inner_filters=256, filters=[64, 256, 512, 1024, 2048])
# up = UpsamplingBilinear2d(scale_factor=4)
# final = Conv1x1(in_channels=128, out_channels=1)
# dropout = nn.Dropout2d(p=0.15)
# _initialize_weights(fpn)
# _initialize_weights(final)

# testmod = SCSeResneXt(seg_classes=1, backbone_arch='seresnext101')
# testmod.encoder_stages[0]
# testconv0 = testmod.encoder_stages[0]
# testconv1 = testmod.encoder_stages[1]
# testconv2 = testmod.encoder_stages[2]
# testconv3 = testmod.encoder_stages[3]
# testconv4 = testmod.encoder_stages[4]
# output0 = testconv0(input)
# output1 = testconv1(output0)
# output2 = testconv2(output1)
# output3 = testconv3(output2)
# output4 = testconv4(output3)
# output4.shape
# testmod.forward(input)
# testmod.decoder_stages
# img = np.zeros((512, 512, 3), dtype='int')

# enc_results = []
# # x = fpn
# torch.nn.Conv2d
# stage0 = testmod.encoder_stages[0]
# stage1 = testmod.encoder_stages[1]
# stage2 = testmod.encoder_stages[2]
# stage3 = testmod.encoder_stages[3]
# stage4 = testmod.encoder_stages[4]

############################################################################################################

# testmod = models.unet.SCSeResneXt(seg_classes=1, backbone_arch='seresnext101')
# testmod.encoder_stages[0]

# # x = stage0.forward(input)
# enc_results = []
# x = input
# for i, stage in enumerate(testmod.encoder_stages):
#     x = stage(x)
#     if i > 0:
#         enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
# seg = fpn(list(reversed(enc_results)))
# seg = up(seg)
# finalout = final(seg)
# finalout.shape
# #     # enc_results.append(x.clone()) # from unet.SCSeResNeXt
# # x = torch.as_tensor(img)
# # for stage in testmod.encoder_stages:
# #             x = stage(x)
# #             enc_results.append(x.clone())

# # seg = fpn(list(reversed(enc_results)))
# # seg = up(seg)
# # seg = dropout(seg)
# # x = final(seg)
# op = testmod.encoder_stages[1]
# op(input).shape

# testmod = models.unet.SCSeResneXt(seg_classes=1, backbone_arch='seresnext101')
# fpn = FPNSegmentation(inner_filters=256, filters=models.unet.encoder_params['seresnext101']["filters"])
    
# x = torch.rand(5, 3, 512, 512)
# enc_results = []
# for i, stage in enumerate(testmod.encoder_stages):
#     x = stage(x)
#     if i > 0:
#         enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

# up = UpsamplingBilinear2d(scale_factor=4)
# conv_skip = Conv3x3(in_channels=256, out_channels=128)
# conv = Conv3x3(in_channels=128, out_channels=128)
# conv_post_1 = Conv3x3(in_channels=128, out_channels=128)
# conv_post_2 = Conv3x3(in_channels=128, out_channels=64)
# final = Conv1x1(in_channels=64, out_channels=1)
# dropout = nn.Dropout2d(p=0.15)
# _initialize_weights(fpn)
# _initialize_weights(final)

# seg = fpn(list(reversed(enc_results)))                                 # 128
# seg = conv(seg)                                    # added from dsb    # 128
# skip = conv_skip(enc_results[0])                   # added from dsb    # 128 to reduce skip from 256 to 128
# seg = torch.cat([seg, skip], dim = 1)              # added from dsb    # 128
# seg.shape
# skip.shape
# seg = conv_skip(seg)                               # added from dsb    # 128
# seg.shape
# seg = up(seg)
# seg = dropout(seg)
# seg = conv_post_1(seg)                             # added from dsb    # 64
# seg = conv_post_2(seg)                             # added from dsb    # 64
# x = final(seg)                
# x.shape