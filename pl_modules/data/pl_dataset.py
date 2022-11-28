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
import math

import kornia as K


class MultiTransform(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()

        self.transforms = []
        for t in transforms:
            if isinstance(t, list):
                t = K.augmentation.container.AugmentationSequential(*t)
            self.transforms.append(t)
        self.transforms = torch.nn.ModuleList(self.transforms)

    def forward(self, *args, only_first=False, **kwargs):
        outs = []
        for i, t in enumerate(self.transforms):
            outs.append(t(*args, **kwargs))
            if i == 0 and only_first:
                break
        outs = torch.cat(outs, dim=-3)
        return outs


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
            colors = self.opt['color'].split(',')
            for color in colors:
                if color == 'rgb':
                    pipeline.append([torch.nn.Identity()])
                    pipeline_inv.append([torch.nn.Identity()])
                elif color == 'rg':
                    pipeline.append([K.contrib.Lambda(lambda x: x[:, :2])])
                    pipeline_inv.append([torch.nn.Identity()])
                elif color == 'gb':
                    pipeline.append([K.contrib.Lambda(lambda x: x[:, 1:])])
                    pipeline_inv.append([torch.nn.Identity()])
                elif color == 'rb':
                    pipeline.append([K.contrib.Lambda(lambda x: torch.cat((x[:, 0:1], x[:, 2:]), dim=1))])
                    pipeline_inv.append([torch.nn.Identity()])
                elif color == 'rgb_interpolated':
                    n_channels = 3
                    
                    def interpolate_color(x):
                        x_new = torch.zeros(x.shape[0], 3*n_channels, *x.shape[2:], device=x.device)
                        for i in range(x.shape[1]):
                            for j in range(n_channels):
                                v_min = j / n_channels
                                v_max = (j+1) / n_channels if j < n_channels-1 else 1.0
                                mask = (x[:, i] < v_min) | (x[:, i] > v_max)
                                x_new[:, i*3+j] = (x[:, i] - v_min) / (v_max - v_min)
                                x_new[:, i*3+j][mask] = -1
                        return x_new
                        
                    def deinterpolate_color(x):
                        x_new = torch.zeros(x.shape[0], 3, *x.shape[2:], device=x.device)
                        for i in range(x.shape[1]):
                            j = i // n_channels
                            k = i % n_channels
                            v_min = k / n_channels
                            v_max =(k+1) / n_channels if k < n_channels-1 else 1.0
                            mask = x[:, i] >= 0
                            x_new[:, j][mask] = x[:, i][mask] * (v_max - v_min) + v_min
                        return x_new
                        
                    pipeline.append([K.contrib.Lambda(interpolate_color)])
                    pipeline_inv.append([K.contrib.Lambda(deinterpolate_color)])
                elif color == 'y':
                    pipeline.append([K.color.RgbToYcbcr(),
                                     K.contrib.Lambda(lambda x: x[:, :1])])
                    pipeline_inv.append([K.color.YcbcrToRgb()])
                elif color == 'ycbcr':
                    pipeline.append([K.color.RgbToYcbcr()])
                    pipeline_inv.append([K.color.YcbcrToRgb()])
                elif color == 'hsv':
                    def hsv_norm_func(x):
                        x = x.clone()
                        x[:, 0] /= 2*math.pi
                        return x
                    def hsv_denorm_func(x):
                        x = x.clone()
                        x[:, 0] *= 2*math.pi
                        return x
                        
                    pipeline.append([K.color.RgbToHsv(), 
                                     K.contrib.Lambda(hsv_norm_func)])
                    pipeline_inv.append([K.contrib.Lambda(hsv_denorm_func),
                                         K.color.HsvToRgb()])
                elif color == 'hls':
                    def hsv_norm_func(x):
                        x = x.clone()
                        x[:, 0] /= 2*math.pi
                        return x
                    def hsv_denorm_func(x):
                        x = x.clone()
                        x[:, 0] *= 2*math.pi
                        return x
                        
                    pipeline.append([K.color.RgbToHls(), 
                                     K.contrib.Lambda(hsv_norm_func)])
                    pipeline_inv.append([K.contrib.Lambda(hsv_denorm_func),
                                         K.color.HlsToRgb()])
                elif color == 'lab':
                    pipeline.append([K.color.RgbToLab()])
                    pipeline_inv.append([K.color.LabToRgb()])
                elif color == 'luv':
                    pipeline.append([K.color.RgbToLuv()])
                    pipeline_inv.append([K.color.LuvToRgb()])
                elif color == 'yuv':
                    pipeline.append([K.color.RgbToYuv()])
                    pipeline_inv.append([K.color.YuvToRgb()])
                elif color == 'xyz':
                    pipeline.append([K.color.RgbToXyz()])
                    pipeline_inv.append([K.color.XyzToRgb()])
                elif color == 'oklab':
                    pipeline.append([RgbToOklab()])
                    pipeline_inv.append([OklabToRgb()])
                elif color == 'hsi':
                    def hsv_norm_func(x):
                        x = x.clone()
                        x[:, 0] /= 2*math.pi
                        return x
                    def hsv_denorm_func(x):
                        x = x.clone()
                        x[:, 0] *= 2*math.pi
                        return x
                        
                    pipeline.append([RgbToHsi(), 
                                     K.contrib.Lambda(hsv_norm_func)])
                    pipeline_inv.append([K.contrib.Lambda(hsv_denorm_func),
                                         HsiToRgb()])
                else:
                    raise NotImplementedError()

                mean = self.opt.get(f'mean_{color}', None)
                std = self.opt.get(f'std_{color}', None)
                if mean is not None and std is not None:
                    pipeline[-1].append(K.enhance.Normalize(mean, std))
                    pipeline_inv[-1].insert(0, K.enhance.Denormalize(mean, std))

        if pipeline:
            self.pipeline = MultiTransform(pipeline)
            pipeline_inv = pipeline_inv[0]
            #self.pipeline = K.augmentation.container.AugmentationSequential(*pipeline)
            self.pipeline_inv = K.augmentation.container.AugmentationSequential(*pipeline_inv)
        else:
            self.pipeline = None
            self.pipeline_inv = None

    def apply_transform(self, batch):
        if self.pipeline is not None:
            batch['lq_org'] = batch['lq']
            batch['lq'] = self.pipeline(batch['lq'].to(torch.float32)).to(batch['lq'].dtype)

            if 'gt' in batch:
                batch['gt_org'] = batch['gt']
                batch['gt'] = self.pipeline(batch['gt'].to(torch.float32), only_first=True).to(batch['gt'].dtype)
        else:
            batch['lq_org'] = batch['lq']
            if 'gt' in batch:
                batch['gt_org'] = batch['gt']
        return batch

    def apply_inv_transform(self, batch):
        assert 'out' in batch
        
        if self.pipeline_inv is not None:
            colors = self.opt['color'].split(',')
            if colors[0] == 'y':
                lq = F.interpolate(batch['lq_org'], size=batch['out'].shape[-2:], mode='bicubic')
                lq = K.color.rgb_to_ycbcr(lq)
                batch['out'] = torch.cat((batch['out'], lq[:, 1:]), dim=1)
            elif colors[0] == 'rg':
                lq = F.interpolate(batch['lq_org'], size=batch['out'].shape[-2:], mode='bicubic')
                batch['out'] = torch.cat((batch['out'], lq[:, 2:]), dim=1)
            elif colors[0] == 'gb':
                lq = F.interpolate(batch['lq_org'], size=batch['out'].shape[-2:], mode='bicubic')
                batch['out'] = torch.cat((lq[:, 0:1], batch['out']), dim=1)
            elif colors[0] == 'rb':
                lq = F.interpolate(batch['lq_org'], size=batch['out'].shape[-2:], mode='bicubic')
                batch['out'] = torch.cat((batch['out'][:, 0:1], lq[:, 1:2], batch['out'][:, 1:]), dim=1)
            batch['out'] = self.pipeline_inv(batch['out'].to(torch.float32)).to(batch['out'].dtype)
            batch['lq'] = batch['lq_org']
            if 'gt' in batch:
                batch['gt'] = batch['gt_org']
        return batch


