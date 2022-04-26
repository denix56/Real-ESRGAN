import logging
import os.path as osp

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from basicsr.train import parse_options, copy_opt_file
from basicsr.utils import make_exp_dirs, scandir

from pl_modules.callbacks.ema import EMA
from pl_modules.models import build_model
from pl_modules.data.pl_dataset import PLDataset


def find_resume_ckpt(opt):
    if opt['auto_resume']:
        model_path = opt['path']['models']
        if osp.isdir(model_path):
            ckpts = list(scandir(model_path, suffix='ckpt', recursive=False, full_path=False))
            if len(ckpts) != 0:
                resume_ckpt = None
                for v in ckpts:
                    if v == 'last.ckpt':
                        resume_ckpt = 'last.ckpt'
                if not resume_ckpt:
                    resume_step = 0
                    for v in ckpts:
                        step = v[:-5].split('-')[-1]
                        if step > resume_step:
                            resume_ckpt = v
                            resume_step = step
                if resume_ckpt:
                    resume_ckpts_path = osp.join(model_path, resume_ckpt)
                    opt['path']['resume_ckpt'] = resume_ckpts_path
                    return resume_ckpts_path
    return None


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    resume_ckpt_path = find_resume_ckpt(opt)

    pl.seed_everything(opt['manual_seed'], workers=True)

    total_iters = int(opt['train']['total_iter'])

    data = PLDataset(opt)
    model = build_model(opt)

    ms = ModelSummary(-1)
    mcp = ModelCheckpoint(opt['path']['models'], monitor='val/PeakSignalNoiseRatio', mode='max', save_last=True)
    ema = EMA(opt['train']['ema_decay'])

    if opt['logger'].get('use_tb_logger'):
        logger = TensorBoardLogger(osp.join(opt['root_path'], 'tb_logger'))
    else:
        logger = None

    deterministic = False

    trainer = pl.Trainer(logger=logger, callbacks=[ms, mcp, ema], devices=opt['num_gpu'], accelerator='gpu',
                         max_steps=total_iters, benchmark=True, deterministic=deterministic,
                         precision=16 if opt['train'].get('mixed') else 32, strategy='ddp', fast_dev_run=False)
    if deterministic:
        # We have bilinear interpolation somewhere
        torch.use_deterministic_algorithms(True, warn_only=True)

    if resume_ckpt_path is None and trainer.is_global_zero:
        make_exp_dirs(opt)
        copy_opt_file(args.opt, opt['path']['experiments_root'])

    trainer.fit(model, datamodule=data, ckpt_path=resume_ckpt_path)
