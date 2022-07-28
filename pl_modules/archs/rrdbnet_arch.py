from pl_modules import PL_ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights

from torch import nn
import torch.nn.functional as F
import torch


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.ModuleList(layers)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, alpha_p=False):
        super(ResidualDenseBlock, self).__init__()
        act_func = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)

        num_interm_layers = 4

        self.layers = nn.ModuleList(
            [nn.ModuleList([nn.Conv2d(num_feat + i * num_grow_ch, num_grow_ch, 3, 1, 1),
             act_func()]) for i in range(num_interm_layers)]
        )
        self.last_conv = nn.Conv2d(num_feat + num_interm_layers * num_grow_ch, num_feat, 3, 1, 1)
        # initialization
        default_init_weights([self.layers, self.last_conv], 0.1)
        
        self.alpha = nn.Parameter(torch.tensor([0.2]), requires_grad=alpha_p)

    def forward(self, x, return_all=False):
        inp = x
        outs = None
        if return_all:
            outs = []
        for layer in self.layers:
            l_out = x
            for i, layer2 in enumerate(layer):
                l_out = layer2(l_out)
                if i == 0 and return_all:
                    outs.append(l_out)
            x = torch.cat((x, l_out), dim=1)
        x = self.last_conv(x)
        if return_all:
            outs.append(x)
            
        return x * self.alpha + inp, outs


class RRDB(nn.Module):
    """Residual in Residual Dense Block.
    Used in RRDB-Net in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32, use_attn=False, alpha_p=False):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, alpha_p=alpha_p)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, alpha_p=alpha_p)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, alpha_p=alpha_p)

        if use_attn:
            self.attn = nn.MultiheadAttention(num_feat*16, 8, batch_first=True)
        else:
            self.attn = None
            
        self.alpha = nn.Parameter(torch.tensor([0.2]), requires_grad=alpha_p)

    def forward(self, x, return_all=False):
        out, outs = self.rdb1(x, return_all)
        out, outs2 = self.rdb2(out, return_all)
        out, outs3 = self.rdb3(out, return_all)
        
        if return_all:
            outs = outs + outs2 + outs3
        
        if self.attn:
            out = F.pixel_unshuffle(out, 4)
            shape = out.shape
            out = out.view(*shape[:2], -1).permute(0, 2, 1)
            out, _ = self.attn(out, out, out, need_weights=False)
            out = out.permute(0, 2, 1).reshape(shape)
            out = F.pixel_shuffle(out, 4)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * self.alpha + x, outs


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

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, 
    ret_all=False, use_attn=False, add_feat=False, add_x=False, alpha_p=False, shuffle=False):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.upsample_mode = 'bilinear'
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
            
        act_func = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch, use_attn=use_attn, alpha_p=alpha_p)
        if shuffle:
            assert not add_feat
            self.conv_body = nn.Sequential(
                                 nn.Conv2d(num_feat, num_feat*2, 3, 1, 1),
                                 act_func(),
                                 nn.Conv2d(num_feat*2, num_feat*2, 3, 1, 1),
                                 act_func()
                             )
            self.conv_up1 = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat//2, num_feat//2, 3, 1, 1),
                act_func(),
                nn.Conv2d(num_feat//2, num_feat//2, 3, 1, 1),
                act_func(),
            )
            self.conv_up2 = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat//8, num_feat//8, 3, 1, 1),
                act_func(),
                nn.Conv2d(num_feat//8, num_feat//8, 3, 1, 1),
                act_func(),
            )
            
            self.torgb1 = nn.Conv2d(num_feat*2, num_out_ch, 3, 1, 1)
            self.torgb2 = nn.Conv2d(num_feat//2, num_out_ch, 3, 1, 1)
            self.torgb3 = nn.Conv2d(num_feat//8, num_out_ch, 3, 1, 1)

            
        else:
            self.conv_body = nn.Sequential(
                                 nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                             )

            # upsample
            self.conv_up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=self.upsample_mode),
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

        self.ret_all = ret_all
        self.add_feat = add_feat
        self.add_x = add_x

    def forward(self, x, ret_all=False, pad=False):
        if pad:
            h, w = x.shape[-2:]
            h_pad, w_pad = h, w
            if h_pad % 4 != 0:
                h_pad = (h // 4 + 1) * 4
            if w_pad % 4 != 0:
                w_pad = (w // 4 + 1) * 4

            left = (w_pad - w) // 2
            right = w_pad - w - left
            top = (h_pad - h) // 2
            bottom = h_pad - h - top

            x = F.pad(x, (left, right, top, bottom))

        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_out = self.body(feat)
        body_feat = self.conv_body(body_out)
        if self.add_feat:
            feat = feat + body_feat
        else:
            feat = body_feat
        # upsample
        if self.add_x:   
            rgb1 = x + self.torgb1(feat)
        else:
            rgb1 = self.torgb1(feat)
        rgb = F.interpolate(rgb1, scale_factor=2, mode=self.upsample_mode)
        feat = self.conv_up1(feat)
        rgb2 = rgb + self.torgb2(feat)
        rgb = F.interpolate(rgb2, scale_factor=2, mode=self.upsample_mode)
        feat = self.conv_up2(feat)
        rgb3 = rgb + self.torgb3(feat)
        
        if pad:
            top = top*self.scale
            left = left * self.scale
            bottom = -bottom*self.scale if bottom > 0 else None
            right = -right*self.scale if right > 0 else None

            rgb3 = rgb3[..., top:bottom, left:right]

        if self.ret_all or ret_all:
            return [rgb1, rgb2, rgb3]
        else:
            return rgb3
       
       
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

    def forward(self, x, return_all=False):
        outs = None
        if return_all:
            outs = []
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        if return_all:
            outs.append(feat)
        body_feat = feat
        for l in self.body:
            body_feat, outs2 = l(body_feat, return_all)
            if return_all:
                outs = outs + outs2
        body_feat = self.conv_body(body_feat)
        if return_all:
            outs.append(body_feat)
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out, outs
