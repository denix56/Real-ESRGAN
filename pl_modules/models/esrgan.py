import torch

from pl_modules.registry import PL_MODEL_REGISTRY
from pl_modules.models.pl_srgan_model import SRGANModel


@PL_MODEL_REGISTRY.register()
class ESRGANModel(SRGANModel):
    """ESRGAN model for single image super-resolution."""

    def training_step(self, batch, batch_idx, optimizer_idx):
        lq = batch['lq']
        gt = batch.get('gt')
        l1_gt = batch.get('l1_gt')
        percep_gt = batch.get('percep_gt')
        gan_gt = batch.get('gan_gt')

        if optimizer_idx == 0:
            output = self.net_g(lq)
            l_g_total = 0
            loss_dict = {}
            if self.cri_pix:
                l_g_pix = self.cri_pix(output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            real_d_pred = self.net_d(gan_gt).detach()
            fake_g_pred = self.net_d(output)
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            loss_dict['l_g_total'] = l_g_total
            self.log_dict(loss_dict)

            return l_g_total
        elif optimizer_idx == 1:
            l_d_total = 0
            loss_dict = {}

            # gan loss (relativistic gan)

            # In order to avoid the error in distributed training:
            # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
            # the variables needed for gradient computation has been modified by
            # an inplace operation",
            # we separate the backwards for real and fake, and also detach the
            # tensor for calculating mean.

            # real
            output = self.net_g(lq)
            fake_d_pred = self.net_d(output).detach()
            real_d_pred = self.net_d(gt)
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5

            # fake
            fake_d_pred = self.net_d(output.detach())
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5

            l_d_total = l_d_real + l_d_fake

            loss_dict['l_d_real'] = l_d_real
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            self.log_dict(loss_dict)

            if self.global_step % 500 == 0:
                self._save_images(lq, output, gt, prefix='train')

            return l_d_total
