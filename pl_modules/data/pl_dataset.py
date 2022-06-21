from typing import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pl_bolts.datamodules import AsynchronousLoader
from basicsr.data import build_dataset
import torch
import torch.nn.functional as F
import pl_modules.data
from pl_modules.data.augment import RgbToHsi, HsiToRgb, RgbToOklab, OklabToRgb
from copy import deepcopy

import kornia as K


class PLDataset(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.train_ds = None
        self.val_ds = None

        self.batch_size_per_gpu = None
        self.num_worker_per_gpu = None

        self.val_ds = []
        for phase, dataset_opt in self.opt['datasets'].items():
            if phase == 'train':
                dataset_opt['mode'] = 'train'
                self.train_ds = (build_dataset(dataset_opt), dataset_opt)
                if hasattr(self.train_ds[0], 'setup'):
                    self.train_ds[0].setup()
            elif phase.split('_')[0] == 'val':
                dataset_opt['mode'] = 'val'
                self.val_ds.append((build_dataset(dataset_opt), dataset_opt))
                if hasattr(self.val_ds[-1][0], 'setup'):
                    self.val_ds[-1][0].setup()
            else:
                raise ValueError(f'Dataset phase {phase} is not recognized.')

        self._create_transform()

    def train_dataloader(self):
        ds = self.train_ds[0]
        dataset_opt = self.train_ds[1]
        loader = DataLoader(ds, batch_size=dataset_opt['batch_size_per_gpu'], shuffle=dataset_opt['use_shuffle'],
                            num_workers=dataset_opt['num_worker_per_gpu'], drop_last=True,
                            pin_memory=dataset_opt.get('pin_memory', False),
                            persistent_workers=(dataset_opt.get('persistent_workers', False)) and dataset_opt['num_worker_per_gpu'] > 0)
        if not self.opt.get('preprocess_cpu', False) and self.trainer.num_devices == 1:
            loader = AsynchronousLoader(loader)
        return loader

    def val_dataloader(self):
        loaders = []
        for ds, dataset_opt in self.val_ds:
            loader = DataLoader(ds, batch_size=1, shuffle=dataset_opt.get('use_shuffle_test', False),
                                num_workers=0, drop_last=False,
                                pin_memory=dataset_opt.get('pin_memory', True))
            loaders.append(loader)
        if len(loaders) == 1:
            loaders = loaders[0]
        return loaders

    def _create_transform(self):
        pipeline = []
        pipeline_inv = []

        if 'color' in self.opt:
            if self.opt['color'] == 'rgb':
                pass
            elif self.opt['color'] == 'y':
                pipeline.append(K.color.RgbToYcbcr())
                pipeline_inv.append(K.color.YcbcrToRgb())
                pipeline.append(K.contrib.Lambda(lambda x: x[:, :1]))
            elif self.opt['color'] == 'ycbcr':
                pipeline.append(K.color.RgbToYcbcr())
                pipeline_inv.append(K.color.YcbcrToRgb())
            elif self.opt['color'] == 'hsv':
                pipeline.append(K.color.RgbToHsv())
                pipeline_inv.append(K.color.HsvToRgb())
            elif self.opt['color'] == 'hls':
                pipeline.append(K.color.RgbToHls())
                pipeline_inv.append(K.color.HlsToRgb())
            elif self.opt['color'] == 'lab':
                pipeline.append(K.color.RgbToLab())
                pipeline_inv.append(K.color.LabToRgb())
            elif self.opt['color'] == 'luv':
                pipeline.append(K.color.RgbToLuv())
                pipeline_inv.append(K.color.LuvToRgb())
            elif self.opt['color'] == 'yuv':
                pipeline.append(K.color.RgbToYuv())
                pipeline_inv.append(K.color.YuvToRgb())
            elif self.opt['color'] == 'xyz':
                pipeline.append(K.color.RgbToXyz())
                pipeline_inv.append(K.color.XyzToRgb())
            elif self.opt['color'] == 'oklab':
                pipeline.append(RgbToOklab())
                pipeline_inv.append(OklabToRgb())
            elif self.opt['color'] == 'hsi':
                pipeline.append(RgbToHsi())
                pipeline_inv.append(HsiToRgb())
            else:
                raise NotImplementedError()

        mean = self.opt['mean'] if 'mean' in self.opt else None
        std = self.opt['std'] if 'std' in self.opt else None
        if mean is not None and std is not None:
            pipeline.append(K.enhance.Normalize(mean, std))
            pipeline_inv.append(K.enhance.Denormalize(mean, std))

        if pipeline:
            self.pipeline = K.augmentation.container.AugmentationSequential(*pipeline)
            self.pipeline_inv = K.augmentation.container.AugmentationSequential(*pipeline_inv[::-1])
        else:
            self.pipeline = None
            self.pipeline_inv = None

    def apply_transform(self, batch):
        if self.pipeline is not None:
            dtype = batch['lq'].dtype
            batch['lq_org'] = batch['lq']
            batch['lq'] = self.pipeline(batch['lq'].to(torch.float32)).to(dtype)
            

            if 'gt' in batch:
                dtype = batch['gt'].dtype
                batch['gt_org'] = batch['gt']
                batch['gt'] = self.pipeline(batch['gt'].to(torch.float32)).to(dtype)
        else:
            batch['lq_org'] = batch['lq']
            if 'gt' in batch:
                batch['gt_org'] = batch['gt']
        return batch

    def apply_inv_transform(self, batch):
        assert 'out' in batch
        
        if self.pipeline_inv is not None:
            if self.opt['color'] == 'y':
                lq = F.interpolate(batch['lq_org'], size=batch['out'].shape[-2:], mode='bicubic')
                lq = K.color.rgb_to_ycbcr(lq)
                batch['out'] = torch.cat((batch['out'], lq[:, 1:]), dim=1)
            dtype = batch['out'].dtype
            batch['out'] = self.pipeline_inv(batch['out'].to(torch.float32)).to(dtype)
            batch['lq'] = batch['lq_org']
            if 'gt' in batch:
                batch['gt'] = batch['gt_org']
        return batch


