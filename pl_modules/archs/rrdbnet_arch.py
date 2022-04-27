from pl_modules import PL_ARCH_REGISTRY
from basicsr.archs.arch_util import make_layer
from basicsr.archs.rrdbnet_arch import RRDB

from torch import nn
import torch.nn.functional as F


@PL_ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.torgb1 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.torgb2 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.torgb3 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        rgb = F.interpolate(self.torgb1(feat), scale_factor=2, mode='bilinear')
        feat = self.conv_up1(feat)
        rgb = F.interpolate(rgb + self.torgb2(feat), scale_factor=2, mode='bilinear')
        feat = self.conv_up2(feat)
        out = rgb + self.torgb3(feat)
        return out
