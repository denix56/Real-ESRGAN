import torch
from basicsr.models import lr_scheduler as lr_scheduler
from torch import optim
import pytorch_lightning as pl
from abc import ABC, abstractmethod


class BaseModel(ABC, pl.LightningModule):
    """PL Base model."""

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.preprocess_cpu = opt.get('preprocess_cpu', False)
        self.save_hyperparameters(opt)

    @abstractmethod
    def _setup_optimizers(self):
        pass

    def _setup_schedulers(self, optimizers):
        """Set up schedulers."""
        schedulers = []
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        warmup_iter = self.opt['train'].get('warmup_iter', -1)

        for optimizer in optimizers:
            if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
                sched = lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler'])
            elif scheduler_type == 'CosineAnnealingRestartLR':
                sched = lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler'])
            else:
                raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
            if warmup_iter > 0:
                warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=1. / warmup_iter, total_iters=warmup_iter)
                sched = optim.lr_scheduler.ChainedScheduler([sched, warmup_sched])
            schedulers.append(sched)
        return schedulers

    @abstractmethod
    def _feed_data(self, data):
        pass

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.preprocess_cpu:
            batch = self._feed_data(batch)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if not self.preprocess_cpu:
            batch = self._feed_data(batch)
        return batch

    def configure_optimizers(self):
        optimizers = self._setup_optimizers()
        schedulers = self._setup_schedulers(optimizers)
        return optimizers, schedulers

    def _get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


