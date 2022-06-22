# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

from pl_modules.models.pl_sr_model import SRModel

from pl_modules import PL_MODEL_REGISTRY


@PL_MODEL_REGISTRY.register()
class SwinIRModel(SRModel):
    def __init__(self, opt):
        super().__init__(opt)




# if __name__ == '__main__':
#     upscale = 4
#     window_size = 8
#     height = (1024 // upscale // window_size + 1) * window_size
#     width = (720 // upscale // window_size + 1) * window_size
#     model = SwinIR(upscale=2, img_size=(height, width),
#                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
#                    embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
#     print(model)
#     print(height, width, model.flops() / 1e9)
#
#     x = torch.randn((1, 3, height, width))
#     x = model(x)
#     print(x.shape)
