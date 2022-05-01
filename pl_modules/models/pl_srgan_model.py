from copy import deepcopy

import torch
import torch.nn.functional as F
from pl_modules import PL_MODEL_REGISTRY
from pl_modules.archs import build_network
from basicsr.losses import build_loss
from overrides import overrides
from pl_modules.models.pl_sr_model import SRModel


@PL_MODEL_REGISTRY.register()
class SRGANModel(SRModel):
    """SRGAN model for single image super-resolution."""

    def __init__(self, opt):
        super().__init__(opt)
        # define network net_d
        self.cat_imgs = self.opt['train'].get('cat_imgs', False)
        opt_d = deepcopy(self.opt['network_d'])
        if self.cat_imgs:
            opt_d['num_in_ch'] *= 2
        self.net_d = build_network(opt_d)

        train_opt = self.opt['train']
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

    @overrides
    def _build_losses(self):
        super()._build_losses()

        train_opt = self.opt['train']
        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt'])

    @overrides
    def _setup_optimizers(self):
        train_opt = self.opt['train']
        optimizers = []
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optimizer_g = self._get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        optimizers.append(optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optimizer_d = self._get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        optimizers.append(optimizer_d)
        return optimizers

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = super().on_after_batch_transfer(batch, dataloader_idx)

        gt = batch.get('gt')
        if gt is not None:
            batch['l1_gt'] = gt
            batch['percep_gt'] = gt
            batch['gan_gt'] = gt
        return batch

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
            # gan loss
            if self.cat_imgs:
                lq_scaled = F.interpolate(lq, size=output.shape[-2:], mode='bilinear')
                output = torch.cat((lq_scaled, output), dim=1)
            fake_g_pred = self.net_d(output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            loss_dict['l_g_total'] = l_g_total
            self.log_dict(loss_dict)

            return l_g_total
        elif optimizer_idx == 1:
            l_d_total = 0
            loss_dict = {}

            output = self.net_g(lq)
            output_org = output

            if self.cat_imgs:
                lq_scaled = F.interpolate(lq, size=gan_gt.shape[-2:], mode='bilinear')
                gan_gt = torch.cat((lq_scaled, gan_gt), dim=1)
                output = torch.cat((lq_scaled, output), dim=1)

            real_d_pred = self.net_d(gan_gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_total += l_d_real
            # fake
            fake_d_pred = self.net_d(output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_total += l_d_fake
            self.log_dict(loss_dict)

            if self.global_step % 250 == 0:
                self._save_images(lq, output_org, gt, prefix='train')

            return l_d_total

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        if optimizer_idx == 0:
            if self.global_step > self.net_d_init_iters and (batch_idx + 1) % self.net_d_iters == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()
        elif optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)
            
    def _load_weights(self):
        super()._load_weights()
        self._load_network(self.net_d, 'd')
