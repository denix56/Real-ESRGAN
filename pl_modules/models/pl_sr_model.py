import itertools
import os
import os.path as osp
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import save_image
from torchmetrics import MetricCollection
from pl_modules.models.pl_base_model import BaseModel
from pl_modules import PL_MODEL_REGISTRY
from pl_modules.archs import build_network
from basicsr.losses import build_loss
from overrides import overrides
from pl_modules.metrics import build_metric
from pl_modules.metrics.metric_util import GatherImages

import torch.nn as nn

import numpy as np


def inverse_leaky_relu(x, negative_slope=1e-2, inplace=False):
    return F.leaky_relu(x, 1./negative_slope, inplace=inplace)


class InverseLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 1e-2, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inlace = inplace

    def forward(self, x):
        return inverse_leaky_relu(x, self.negative_slope, inplace=self.inplace)


class ATanh(nn.Module):
    def __init__(self, eps=1e-8):
        super(ATanh, self).__init__()
        self.eps = eps

    def forward(self, x):
        x = x.clamp(-1+self.eps, 1-self.eps)
        return torch.atanh(x)


class ColorTransform(nn.Module):
    def __init__(self, negative_slope:float = 0.1, eps=1e-6):
        super(ColorTransform, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(3, 32, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(32, 32, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(32, 3, bias=False)
            )
            
        self.rev_net = nn.Sequential(
                nn.Linear(3, 32, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(32, 32, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(32, 3, bias=False)
            )

    def forward(self, x, inv:bool = False, freeze:bool=False):
        x = x.permute(0, 2, 3, 1)
        model = None
        if inv:
            model = self.rev_net
        else:
            model = self.net
            
        if freeze:
            model = deepcopy(model)
            for param in model.parameters():
                param.requires_grad = False
            
        x = model(x)
        x = x.permute(0, 3, 1, 2)
        return x
        
        
class RandomColorSpace(nn.Module):
    def __init__(self, n_color_spaces = 1):
        super().__init__()
        
        self.n_color_spaces = n_color_spaces
        
        self.net = nn.Sequential(nn.Conv2d(3+self.n_color_spaces, 32, 3, 1, 1),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Conv2d(32, 32, 3, 1, 1))
                                 
    def forward(self, x, idx=None):
        if idx is None:
            idx = np.random.default_rng().choice(self.n_color_spaces)
        x = x[:, idx*3:(idx+1)*3]
        mask = torch.zeros(self.n_color_spaces, device=x.device)
        mask[idx] = 1.0
        mask = mask.view(1, -1, 1, 1)
        shape = list(x.shape)
        shape[1] = -1
        mask = mask.expand(*shape)
        x = torch.cat((x, mask), dim=1)
        
        return self.net(x)
        
        
class ColorTransform2(nn.Module):
    def __init__(self, negative_slope:float = 0.1, eps=1e-6):
        super(ColorTransform, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(4, 32, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(32, 32, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(32, 3, bias=False)
            )

    def forward(self, x, inv:bool = False, freeze:bool=False):
        x = x.permute(0, 2, 3, 1)            
        model = self.net
        
        if inv:
            mask = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        else:
            mask = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        x = torch.cat((x, mask), dim=-1)
            
        if freeze:
            model = deepcopy(model)
            for param in model.parameters():
                param.requires_grad = False
            
        x = model(x)
        x = x.permute(0, 3, 1, 2)
        return x


@PL_MODEL_REGISTRY.register()
class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        opt['network_g']['scale'] = opt['scale']
        self.net_g = build_network(opt['network_g'])

        val_metrics = []
        val_metrics_step = []
        with_metrics = self.opt['val'].get('metrics')
        if with_metrics:
            for name, opt_ in with_metrics.items():
                opt_ = deepcopy(opt_)
                opt_['compute_on_step'] = False
                opt_['input_order'] = 'CHW'
                if opt_['type'] == 'calculate_psnr':
                    opt_['data_range'] = 1.0
                elif opt_['type'] == 'calculate_fid':
                    opt_['reset_real_features'] = False
                elif opt_['type'] == 'calculate_ssim':
                    opt_['compute_on_step'] = True
                if opt_['compute_on_step']:
                    val_metrics_step.append(build_metric(opt_))
                else:
                    val_metrics.append(build_metric(opt_))

        self.val_metrics = MetricCollection(val_metrics, prefix='val/')
        self.val_metrics_step = MetricCollection(val_metrics_step, prefix='val/') if val_metrics_step else None
        self.gather_images = GatherImages()

        if self.opt['train'].get('random_c', False):
            self.random_trans = RandomColorSpace(len(self.opt['color'].split(',')))
        else:
            self.random_trans = None

        train_ds_opt = opt['datasets'].get('train')
        if train_ds_opt:
            gt_size = train_ds_opt['gt_size']
            batch_size = train_ds_opt['batch_size_per_gpu']
            #in_channels = opt['network_g']['num_in_ch']
            scale = opt['scale']
            self.example_input_array = {'lq': torch.zeros(batch_size, 3, gt_size//scale, gt_size//scale)
                                        }

        self.color_transform = lambda x, inv=False, freeze=False: x#ColorTransform()

    def setup(self, stage=None):
        super().setup(stage)
        self._build_losses()
        self._load_weights()

    def _build_losses(self):
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt'])
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt'])
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

    @overrides
    def _setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        if self.random_trans:
            optim_params.extend(self.random_trans.parameters())
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print(f'Params {k} will not be optimized.')
        
        optim_type = train_opt['optim_g'].pop('type')
        optimizer_g = self._get_optimizer(optim_type, 
                                          #itertools.chain(
                                          #self.random_trans,
                                          optim_params, 
                                          #self.color_transform.parameters()), 
                                          **train_opt['optim_g'])
        return [optimizer_g]

    def forward(self, batch):
        lq = batch['lq']
        tmp = self.color_transform(lq)
        if self.random_trans:
            tmp = self.random_trans(tmp, 0)
        tmp = self.net_g(tmp)[0]
        output = self.color_transform(tmp, inv=True)
        return output

    def training_step(self, batch, batch_idx):
        lq = batch['lq']
        gt = batch.get('gt')
        lq_tmp = self.color_transform(lq)
        
        if self.random_trans:
            lq_tmp_t = self.random_trans(lq_tmp)
        else:
            lq_tmp_t = lq_tmp
        output, _ = self.net_g(lq_tmp_t)
        output = self.color_transform(output, inv=True, freeze=True)
        
        lq_output = self.color_transform(lq_tmp.detach(), inv=True)
        l_total = 0
        loss_dict = {}
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(output, gt)
            l_pix_lq = self.cri_pix(lq_output, lq)
            l_total += l_pix + l_pix_lq
            loss_dict['l_pix'] = l_pix
            loss_dict['l_pix_lq'] = l_pix_lq
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(output, gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
                
        loss_dict['l_total'] = l_total
        self.log_dict(loss_dict)

        if self.global_step % 250 == 0:
            with torch.no_grad():
                #output = self.color_transform(output, inv=True)
                if self.trainer.datamodule is not None and hasattr(self.trainer.datamodule, 'apply_inv_transform'):
                    batch['out'] = output
                    batch = self.trainer.datamodule.apply_inv_transform(batch)
                    lq = batch['lq']
                    output = batch['out']
                    gt = batch['gt']
                self._save_images(lq, output, gt, prefix='train')

        return l_total

    def validation_step(self, batch, batch_idx):
        lq = batch['lq']
        gt = batch['gt']
        lq_tmp = self.color_transform(lq)
        
        if self.random_trans:
            lq_tmp = self.random_trans(lq_tmp, 0)
        
        #os.makedirs('saved_feats_hsv', exist_ok=True)
        output, outs = self.net_g(lq_tmp, return_all=False)
        if outs:
            torch.save(outs, f'saved_feats_hsv/{batch_idx}')
        output = self.color_transform(output, inv=True)

        if isinstance(output, list):
            output = output[-1]

        if self.trainer.datamodule is not None and hasattr(self.trainer.datamodule, 'apply_inv_transform'):
            batch['out'] = output
            batch = self.trainer.datamodule.apply_inv_transform(batch)
            lq = batch['lq']
            output = batch['out']
            gt = batch['gt']

        self.val_metrics(output.to(torch.float32), gt.to(torch.float32))
        self.gather_images(lq, output, gt)

    def validation_epoch_end(self, outputs):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()

        lrs, lr_hrs, hrs = self.gather_images.compute()
        self._compute_step_metrics(lr_hrs, hrs)
        self._save_images(lrs, lr_hrs, hrs, prefix='val')
        self.gather_images.reset()

    @rank_zero_only
    @torch.no_grad()
    def _compute_step_metrics(self, preds, gts):
        if self.val_metrics_step is not None:
            metrics = {}
            for i, (pred, gt) in enumerate(zip(preds, gts)):
                metrics_i = self.val_metrics_step(pred.unsqueeze(0).to(torch.float32), gt.unsqueeze(0).to(torch.float32))
                for k, v in metrics_i.items():
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(v)
                self.val_metrics_step.reset()
            for metric, vals in metrics.items():
                metrics[metric] = torch.stack(metrics[metric]).mean()
            self.log_dict(metrics)

    @rank_zero_only
    @torch.no_grad()
    def _save_images(self, lrs, lr_hrs, hrs, prefix='val'):
        path = osp.join(self.opt['path']['visualization'], prefix)
        os.makedirs(path, exist_ok=True)
        is_list_of_tensors = True
        if isinstance(lrs, torch.Tensor) and len(lrs.shape) == 4:
            is_list_of_tensors = False
            lrs = F.interpolate(lrs, size=hrs.shape[-2:], mode='bicubic')

        for i, (lr, lr_hr, hr) in enumerate(zip(lrs, lr_hrs, hrs)):
            if is_list_of_tensors:
                lr = F.interpolate(lr.unsqueeze(0), size=hr.shape[-2:], mode='bicubic').squeeze(0)
            joined = torch.cat((lr, lr_hr, hr), dim=-1).clamp(0, 1)
            self.logger.experiment.add_image(osp.join(prefix, 'image_{}'.format(i)), joined, self.global_step)
            if self.opt['val'].get('save_img', False):
                save_image(joined, osp.join(self.opt['path']['visualization'], prefix, 'image_{}.png'.format(i)))

    def _load_weights(self):
        self._load_network(self.net_g, 'g')

    def _load_network(self, net, postfix):
        path_opt = self.opt['path']
        path = path_opt.get(f'pretrain_network_{postfix}')

        if path is not None:
            strict = path_opt.get(f'strict_load_{postfix}', True)
            cpt = torch.load(path)
            weights = OrderedDict()
            for k, v in cpt['state_dict'].items():
                keys = k.split('.')
                if keys[0] == f'net_{postfix}':
                    weights['.'.join(keys[1:])] = v
            net.load_state_dict(weights, strict=strict)
            print(f'net_{postfix} loaded.')

