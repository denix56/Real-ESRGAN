from copy import deepcopy

import torch
import torch.nn.functional as F

from pl_modules.registry import PL_MODEL_REGISTRY
from pl_modules.models.pl_srgan_model import SRGANModel
from pl_modules.archs import build_network

from overrides import overrides

import kornia as K


@PL_MODEL_REGISTRY.register()
class ESRGANModel(SRGANModel):
    """ESRGAN model for single image super-resolution."""
    def __init__(self, opt):
        super(ESRGANModel, self).__init__(opt)
        opt_d = deepcopy(self.opt['network_d'])
        if self.cat_imgs:
            if 'num_in_ch' in opt_d:
                opt_d['num_in_ch'] *= 2
            elif 'input_nc' in opt_d:
                opt_d['input_nc'] *= 2
        if opt_d['type'] == 'NLayerDiscriminator':
            opt_d['norm_layer'] = torch.nn.InstanceNorm2d

        self.ms = opt.get('multi_scale', False)
        assert self.ms == self.opt['network_g'].get('ret_all', False)

        if self.ms:
            self.net_d = torch.nn.ModuleList([
                build_network(opt_d),
                build_network(opt_d),
                build_network(opt_d)
            ])
        else:
            self.net_d = torch.nn.ModuleList([build_network(opt_d)])

        self.use_l1_gan_loss = self.opt['train'].get('use_l1_gan_loss', False)

    def forward(self, batch):
        lq = batch['lq']
        outputs, _ = self.net_g(lq)
        if not isinstance(outputs, list):
            outputs = [outputs]
        outputs_d = []
        for output, net_d in zip(outputs, self.net_d):
            if self.cat_imgs:
                lq_scaled = F.interpolate(lq, size=output.shape[-2:], mode='bilinear')
                output = torch.cat((lq_scaled, output), dim=1)
            output = net_d(output)
            if isinstance(output, list):
                output = output[-1]
            outputs_d.append(output)
        return outputs_d

    def training_step(self, batch, batch_idx, optimizer_idx):
        lq = batch['lq']
        gt = batch.get('gt')
        l1_gt = batch.get('l1_gt')
        percep_gt = batch.get('percep_gt')
        gan_gt = batch.get('gan_gt')

        if optimizer_idx == 0:
            outputs, _ = self.net_g(lq)
            if not isinstance(outputs, list):
                outputs = [outputs]

            l_g_pix = 0
            l_g_sob = 0
            l_g_percep, l_g_style = 0, 0
            l_g_gan = 0
            loss_dict = {}
            l_g_gan_l1 = 0

            for output, net_d in zip(outputs, self.net_d):
                gan_gt_c = F.interpolate(gan_gt, size=output.shape[-2:], mode='bilinear')
                l1_gt_c = F.interpolate(l1_gt, size=output.shape[-2:], mode='bilinear')
                percep_gt_c = F.interpolate(percep_gt, size=output.shape[-2:], mode='bilinear')

                if self.cri_pix:
                    l_g_pix += self.cri_pix(output, l1_gt_c)
                    #l_g_total += l_g_pix
                    #loss_dict['l_g_pix'] = l_g_pix
                # perceptual loss
                if self.cri_perceptual:
                    if self.trainer.datamodule is not None and hasattr(self.trainer.datamodule, 'apply_inv_transform'):
                        batch['out'] = output
                        batch['gt'] = percep_gt_c
                        batch = self.trainer.datamodule.apply_inv_transform(batch)
                        l_g_percep_c, l_g_style_c = self.cri_perceptual(batch['out'], batch['gt'])
                        batch['gt'] = gt
                        batch['lq'] = lq
                    else:
                        l_g_percep_c, l_g_style_c = self.cri_perceptual(output, percep_gt_c)
                    if l_g_percep_c is not None:
                        l_g_percep += l_g_percep_c
                        #loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style_c is not None:
                        l_g_style += l_g_style_c
                        #loss_dict['l_g_style'] = l_g_style
                if False:
                    d_o = torch.norm(K.filters.spatial_gradient(output, mode='sobel'), dim=2)
                    d_gt = torch.norm(K.filters.spatial_gradient(percep_gt_c, mode='sobel'), dim=2)
                    s_loss = F.l1_loss(d_o, d_gt)
                    l_g_sob += s_loss
                # gan loss (relativistic gan)

                if self.cat_imgs:
                    lq_scaled = F.interpolate(lq, size=gan_gt_c.shape[-2:], mode='bilinear')
                    gan_gt_c = torch.cat((lq_scaled, gan_gt_c), dim=1)
                    output = torch.cat((lq_scaled, output), dim=1)

                real_d_preds = net_d(gan_gt_c)
                fake_g_preds = net_d(output)

                if not isinstance(fake_g_preds, list):
                    fake_g_preds = [fake_g_preds]
                    real_d_preds = [real_d_preds]
                real_d_preds = [real_d_pred.detach() for real_d_pred in real_d_preds]

                for i, (fake_g_pred, real_d_pred) in enumerate(zip(fake_g_preds, real_d_preds)):
                    if i == len(fake_g_preds) - 1:
                        l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
                        l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
                        l_g_gan += (l_g_real + l_g_fake) / 2
                    if self.use_l1_gan_loss:
                        l_g_gan_l1 += F.l1_loss(fake_g_pred, real_d_pred)

            l_g_total = l_g_pix + l_g_percep + l_g_style + l_g_gan + l_g_gan_l1 + l_g_sob
            loss_dict['l_g_pix'] = l_g_pix
            loss_dict['l_g_percep'] = l_g_percep
            loss_dict['l_g_style'] = l_g_style
            loss_dict['l_g_gan'] = l_g_gan
            loss_dict['l_g_gan_l1'] = l_g_gan
            loss_dict['s_loss'] = l_g_sob
            loss_dict['l_g_total'] = l_g_total
            self.log_dict(loss_dict)

            return l_g_total
        elif optimizer_idx == 1:
            loss_dict = {}

            # gan loss (relativistic gan)

            # In order to avoid the error in distributed training:
            # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
            # the variables needed for gradient computation has been modified by
            # an inplace operation",
            # we separate the backwards for real and fake, and also detach the
            # tensor for calculating mean.

            # real
            outputs, _ = self.net_g(lq)
            if not isinstance(outputs, list):
                outputs = [outputs]

            l_d_gan = 0

            output_org = None
            for i, (output, net_d) in enumerate(zip(outputs, self.net_d)):
                output_org = output
                gan_gt_c = F.interpolate(gan_gt, size=output.shape[-2:], mode='bilinear')

                if self.cat_imgs:
                    lq_scaled = F.interpolate(lq, size=gan_gt_c.shape[-2:], mode='bilinear')
                    gan_gt_c = torch.cat((lq_scaled, gan_gt_c), dim=1)
                    output = torch.cat((lq_scaled, output), dim=1)

                fake_d_preds_det = net_d(output)
                real_d_preds = net_d(gan_gt_c)
                fake_d_preds = net_d(output.detach())
                if isinstance(fake_d_preds, list):
                    fake_d_preds_det = fake_d_preds_det[-1]
                    fake_d_preds = fake_d_preds[-1]
                    real_d_preds = real_d_preds[-1]
                fake_d_preds_det = fake_d_preds_det.detach()

                l_d_real = self.cri_gan(real_d_preds - torch.mean(fake_d_preds_det), True, is_disc=True) * 0.5
                # fake
                l_d_fake = self.cri_gan(fake_d_preds - torch.mean(real_d_preds.detach()), False, is_disc=True) * 0.5
                l_d_gan += l_d_real + l_d_fake

                loss_dict[f'l_d_real_{i}'] = l_d_real
                loss_dict[f'l_d_fake_{i}'] = l_d_fake
                loss_dict[f'out_d_real_{i}'] = torch.mean(real_d_preds.detach())
                loss_dict[f'out_d_fake_{i}'] = torch.mean(fake_d_preds.detach())
                loss_dict[f'out_d_fake_{i}'] = torch.mean(fake_d_preds.detach())
            l_d_total = l_d_gan
            loss_dict['l_d_total'] = l_d_total
            self.log_dict(loss_dict)

            if batch_idx % 250 == 0:
                if self.trainer.datamodule is not None and hasattr(self.trainer.datamodule, 'apply_inv_transform'):
                    batch['out'] = output_org
                    batch = self.trainer.datamodule.apply_inv_transform(batch)
                    lq = batch['lq']
                    output_org = batch['out']
                    gt = batch['gt']
                self._save_images(lq, output_org, gt, prefix='train')

            return l_d_total
