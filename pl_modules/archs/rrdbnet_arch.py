from pl_modules import PL_ARCH_REGISTRY
from basicsr.archs.arch_util import make_layer, default_init_weights

from torch import nn
import torch.nn.functional as F
import torch


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        act_func = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)

        num_interm_layers = 4

        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(num_feat + i * num_grow_ch, num_grow_ch, 3, 1, 1), 
            #norm_layer(num_grow_ch), 
            act_func()) for i in range(num_interm_layers)]
        )
        self.last_conv = nn.Conv2d(num_feat + num_interm_layers * num_grow_ch, num_feat, 3, 1, 1)
        # initialization
        default_init_weights([self.layers, self.last_conv], 0.1)

    def forward(self, x):
        inp = x
        for layer in self.layers:
            x = torch.cat((x, layer(x)), dim=1)
        x = self.last_conv(x)
        return x * 0.2 + inp


class RRDB(nn.Module):
    """Residual in Residual Dense Block.
    Used in RRDB-Net in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


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
        self.upsample_mode = 'bilinear'
        #norm_layer = lambda num_feats: nn.InstanceNorm2d(num_feats)

        act_func = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # upsample
        self.conv_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=self.upsample_mode),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            #norm_layer(num_feat),
            act_func(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            #norm_layer(num_feat),
            act_func(),
        )
        self.conv_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=self.upsample_mode),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            #norm_layer(num_feat),
            act_func(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            #norm_layer(num_feat),
            act_func(),
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
        body_out = self.body(feat)
        body_feat = self.conv_body(body_out)
        feat = feat + body_feat
        # upsample
        rgb1 = self.torgb1(feat)
        rgb =  F.interpolate(rgb1, scale_factor=2, mode=self.upsample_mode)
        feat = self.conv_up1(feat)
        rgb2 = rgb + self.torgb2(feat)
        rgb = F.interpolate(rgb2, scale_factor=2, mode=self.upsample_mode)
        feat = self.conv_up2(feat)
        rgb3 = self.torgb3(feat)
        out = rgb + rgb3
        return out
        
        
@PL_ARCH_REGISTRY.register()
class RRDBNet2(nn.Module):
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
        super(RRDBNet2, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample_mode = 'bilinear'

        act_func = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # upsample
        self.conv_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=self.upsample_mode),
            #nn.PixelShuffle(2),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            act_func(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            act_func(),
        )
        self.conv_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=self.upsample_mode),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            act_func(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            act_func(),
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
        body_out = self.body(feat)
        body_feat = self.conv_body(body_out)
        feat = feat + body_feat
        # upsample
        rgb1 = self.torgb1(feat)
        rgb =  F.interpolate(rgb1, scale_factor=2, mode=self.upsample_mode)
        feat = self.conv_up1(feat)
        rgb2 = rgb + self.torgb2(feat)
        rgb = F.interpolate(rgb2, scale_factor=2, mode=self.upsample_mode)
        feat = self.conv_up2(feat)
        out = rgb + self.torgb3(feat)
        return out, rgb1, rgb2
       
       
@PL_ARCH_REGISTRY.register()
class RRDBNetOrg(nn.Module):
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
        super(RRDBNetOrg, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

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
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out