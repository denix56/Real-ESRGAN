import os
import os.path as osp
from copy import deepcopy
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchvision.utils import save_image
from torchmetrics import MetricCollection
from pl_modules.models.pl_base_model import BaseModel
from pl_modules import PL_MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.losses import build_loss
from overrides import overrides
from pl_modules.metrics import build_metric
from pl_modules.metrics.metric_util import GatherImages


@PL_MODEL_REGISTRY.register()
class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.net_g = build_network(opt['network_g'])
        self._build_losses()

        val_metrics = []
        with_metrics = self.opt['val'].get('metrics')
        if with_metrics:
            for name, opt_ in with_metrics.items():
                opt_ = deepcopy(opt_)
                opt_['compute_on_step'] = False
                if name == 'psnr':
                    opt_['data_range'] = 1.0
                    opt_['input_order'] = 'CHW'
                val_metrics.append(build_metric(opt_))

        self.val_metrics = MetricCollection(val_metrics, prefix='val/')
        self.gather_images = GatherImages()

    def _build_losses(self):
        # self.net_g.train()
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
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                self.print(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        optimizer_g = self._get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        return [optimizer_g]

    def training_step(self, batch, batch_idx):
        lq = batch['lq']
        gt = batch.get('gt')

        output = self.net_g(lq)
        l_total = 0
        loss_dict = {}
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(output, gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
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

        if self.global_step % 500 == 0:
            self._save_images(lq, output, gt, prefix='train')

        return l_total

    def validation_step(self, batch, batch_idx):
        lq = batch['lq']
        gt = batch['gt']

        output = self.net_g(lq)
        self.val_metrics(output, gt)
        self.gather_images(lq, output, gt)

    def validation_epoch_end(self, outputs):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()

        lrs, lr_hrs, hrs = self.gather_images.compute()
        self._save_images(lrs, lr_hrs, hrs, prefix='val')
        self.gather_images.reset()

    @rank_zero_only
    @torch.no_grad()
    def _save_images(self, lrs, lr_hrs, hrs, prefix='val'):
        os.makedirs(osp.join(self.opt['path']['visualization'], prefix), exist_ok=True)
        is_list_of_tensors = True
        if isinstance(lrs, torch.Tensor) and len(lrs.shape) == 4:
            is_list_of_tensors = False
            lrs = F.interpolate(lrs, size=hrs.shape[-2:], mode='bicubic')

        for i, (lr, lr_hr, hr) in enumerate(zip(lrs, lr_hrs, hrs)):
            if is_list_of_tensors:
                lr = F.interpolate(lr.unsqueeze(0), size=hr.shape[-2:], mode='bicubic').squeeze(0)
            joined = torch.cat((lr, lr_hr, hr), dim=-1)
            self.logger.experiment.add_image(osp.join(prefix, 'image_{}'.format(i)), joined, self.global_step)
            if self.opt['val'].get('save_img', False):
                save_image(joined, osp.join(self.opt['path']['visualization'], prefix, 'image_{}.png'.format(i)))

