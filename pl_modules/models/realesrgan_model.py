from pl_modules import PL_MODEL_REGISTRY
from pl_modules.models.pl_srgan_model import SRGANModel
from pl_modules.models.realesrnet_model import RealESRNetModel


@PL_MODEL_REGISTRY.register()
class RealESRGANModel(SRGANModel, RealESRNetModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRGANModel, self).__init__(opt)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        super().on_after_batch_transfer(batch, dataloader_idx)

        gt = batch.get('gt')
        gt_usm = batch.get('gt_usm')
        if gt:
            batch['l1_gt'] = gt_usm
            batch['percep_gt'] = gt_usm
            batch['gan_gt'] = gt_usm

            if self.opt['l1_gt_usm'] is False:
                batch['l1_gt'] = gt
            if self.opt['percep_gt_usm'] is False:
                batch['percep_gt'] = gt
            if self.opt['gan_gt_usm'] is False:
                batch['gan_gt'] = gt
