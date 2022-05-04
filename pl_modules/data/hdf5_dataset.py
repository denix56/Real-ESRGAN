import os.path as osp

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class HDF5Dataset(Dataset):
    def __init__(self, path: str, mode:str, mem:bool = False):
        self.path = path
        self.mode = mode
        self.mem = mem
        self.lq = None
        self.gt = None
        self.ds_len = None
        self.file = None

    def setup(self):
        file = h5py.File(self.path, 'r')
        self.ds_len = len(file['{}/lq'.format(self.mode)])

        if self.mem:
            print('Loading data...')
            if self.mode == 'train':
                imgs = file['{}/lq'.format(self.mode)]
                self.lq = np.empty(imgs.shape, dtype=imgs.dtype)
                imgs.read_direct(self.lq)
                self.lq = torch.from_numpy(self.lq)
                #self.lq.share_memory_()

                imgs = file['{}/gt'.format(self.mode)]
                self.gt = np.empty(imgs.shape, dtype=imgs.dtype)
                imgs.read_direct(self.gt)
                self.gt = torch.from_numpy(self.gt)
                #self.gt.share_memory_()

            elif self.mode == 'val':
                imgs_lq = file['{}/lq'.format(self.mode)]
                imgs_gt = file['{}/gt'.format(self.mode)]
                self.lq = []
                self.gt = []
                for (lq_name, lq), (gt_name, gt) in zip(imgs_lq.items(), imgs_gt.items()):
                    lq_arr = np.empty(lq.shape, dtype=lq.dtype)
                    lq.read_direct(lq_arr)
                    lq_arr = torch.from_numpy(lq_arr)
                    gt_arr = np.empty(gt.shape, dtype=gt.dtype)
                    gt.read_direct(gt_arr)
                    gt_arr = torch.from_numpy(gt_arr)

                    #lq_arr.share_memory_()
                    #gt_arr.share_memory_()
                    self.lq.append(lq_arr)
                    self.gt.append(gt_arr)
            else:
                raise NotImplementedError()
        file.close()

    def __getitem__(self, idx):
        data = {}
        if self.mem:
            data['lq'] = self.lq[idx].numpy()
            if self.gt is not None:
                data['gt'] = self.gt[idx].numpy()
        else:
            if self.file is None:
                self.file = h5py.File(self.path, 'r')
            if self.mode != 'train':
                idx = str(idx)
            data['lq'] = self.file['{}/lq'.format(self.mode)][idx][:]
            gt_key = '{}/gt'.format(self.mode)
            if gt_key in self.file:
                data['gt'] = self.file[gt_key][idx][:]
        return data

    def __len__(self):
        return self.ds_len
