from basicsr.train import parse_options, copy_opt_file
from basicsr.data import build_dataset
import realesrgan.data
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pl_modules.callbacks.ema import EMA
from pytorch_lightning.accelerators import GPUAccelerator
import torch
from pl_modules.models import build_model
import logging
import os

from typing import *


from pl_bolts.datamodules import AsynchronousLoader


class PLDataset(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.train_ds = None
        self.val_ds = None

        self.batch_size_per_gpu = None
        self.num_worker_per_gpu = None

    def setup(self, stage: Optional[str] = None):
        self.val_ds = []
        for phase, dataset_opt in self.opt['datasets'].items():
            if phase == 'train':
                self.train_ds = (build_dataset(dataset_opt), dataset_opt)
                self.batch_size_per_gpu = dataset_opt['batch_size_per_gpu']
                self.num_worker_per_gpu = dataset_opt['num_worker_per_gpu']
            elif phase.split('_')[0] == 'val':
                self.val_ds.append((build_dataset(dataset_opt), dataset_opt))
            else:
                raise ValueError(f'Dataset phase {phase} is not recognized.')

    def train_dataloader(self):
        ds = self.train_ds[0]
        dataset_opt = self.train_ds[1]
        loader = DataLoader(ds, batch_size=dataset_opt['batch_size_per_gpu'], shuffle=dataset_opt['use_shuffle'],
                            num_workers=dataset_opt['num_worker_per_gpu'], drop_last=True,
                            pin_memory=dataset_opt.get('pin_memory', False),
                            persistent_workers=dataset_opt.get('persistent_workers', False))
        if not self.opt.get('preprocess_cpu', False) and \
            (self.opt['num_gpu'] == 1 or self.opt['num_gpu'] == 'auto' and GPUAccelerator.auto_device_count() == 1):
            loader = AsynchronousLoader(loader)
        return loader

    def val_dataloader(self):
        loaders = []
        for ds, dataset_opt in self.val_ds:
            loader = DataLoader(ds, batch_size=1, shuffle=False,
                                num_workers=0, drop_last=False,
                                pin_memory=dataset_opt.get('pin_memory', False))
            loaders.append(loader)
        if len(loaders) == 1:
            loaders = loaders[0]
        return loaders


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    pl.seed_everything(opt['manual_seed'], workers=True)

    logger = logging.getLogger('pytorch_lightning')

    total_iters = int(opt['train']['total_iter'])

    data = PLDataset(opt)
    model = build_model(opt)

    ms = ModelSummary(-1)
    mcp = ModelCheckpoint(opt['path']['models'], monitor='val/PeakSignalNoiseRatio', mode='max')
    ema = EMA(opt['train']['ema_decay'])

    if opt['logger'].get('use_tb_logger'):
        logger = TensorBoardLogger(os.path.join(opt['root_path'], 'tb_logger'))
    else:
        logger = None

    trainer = pl.Trainer(logger=logger, callbacks=[ms, mcp, ema], devices=opt['num_gpu'], accelerator='gpu',
                         max_steps=total_iters, benchmark=True, deterministic=True,
                         precision=16 if opt['train'].get('mixed') else 32, fast_dev_run=False)
    torch.use_deterministic_algorithms(True, warn_only=True)
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    # flake8: noqa
    import os.path as osp

    if __name__ == '__main__':
        root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
        train_pipeline(root_path)