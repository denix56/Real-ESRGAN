from typing import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.accelerators import GPUAccelerator
from pl_bolts.datamodules import AsynchronousLoader
from basicsr.data import build_dataset
import data


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
