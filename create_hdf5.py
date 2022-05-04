import h5py
from tqdm.auto import tqdm
import argparse
import glob
import os.path as osp
from PIL import Image
import numpy as np


def process(args):
    with h5py.File(args.output, mode='w') as f:
        train_grp = f.create_group('train')

        paths_lq = sorted(glob.glob(osp.join(args.input_train_lr, '*.png')))
        paths_gt = sorted(glob.glob(osp.join(args.input_train_hr, '*.png')))

        train_lq_ds = None
        train_gt_ds = None

        for i, (f_lq, f_gt) in tqdm(enumerate(zip(paths_lq, paths_gt)), total=len(paths_lq),
                                    desc='Process train images', unit='img'):
            img_lq = np.array(Image.open(f_lq).convert('RGB'))
            if train_lq_ds is None:
                train_lq_ds = train_grp.create_dataset('lq', shape=(len(paths_lq), *img_lq.shape), dtype=img_lq.dtype)
            train_lq_ds[i] = img_lq

            img_gt = np.array(Image.open(f_gt).convert('RGB'))
            if train_gt_ds is None:
                train_gt_ds = train_grp.create_dataset('gt', shape=(len(paths_gt), *img_gt.shape), dtype=img_gt.dtype)
            train_gt_ds[i] = img_gt

        val_grp = f.create_group('val')
        val_lq_grp = val_grp.create_group('lq')
        val_gt_grp = val_grp.create_group('gt')

        paths_lq = sorted(glob.glob(osp.join(args.input_val_lr, '*.png')))
        paths_gt = sorted(glob.glob(osp.join(args.input_val_hr, '*.png')))

        for i, (f_lq, f_gt) in tqdm(enumerate(zip(paths_lq, paths_gt)), total=len(paths_lq),
                                    desc='Process val images', unit='img'):
            img_lq = np.array(Image.open(f_lq).convert('RGB'))
            val_lq_grp.create_dataset(str(i), data=img_lq)

            img_gt = np.array(Image.open(f_gt).convert('RGB'))
            val_gt_grp.create_dataset(str(i), data=img_gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train_hr', type=str, help='Input HR dir')
    parser.add_argument('--input_train_lr', type=str, help='Input LR dir')
    parser.add_argument('--input_val_hr', type=str, help='Input HR dir')
    parser.add_argument('--input_val_lr', type=str, help='Input LR dir')
    parser.add_argument('--output', type=str, help='Output file path')
    args = parser.parse_args()

    process(args)

