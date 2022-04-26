import logging
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from basicsr.train import parse_options, copy_opt_file

from pl_modules.callbacks.ema import EMA
from pl_modules.models import build_model
from pl_modules.data.pl_dataset import PLDataset


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
                         max_steps=total_iters, benchmark=True, deterministic=False,
                         precision=16 if opt['train'].get('mixed') else 32, fast_dev_run=False)
    torch.use_deterministic_algorithms(True, warn_only=True)
    trainer.fit(model, datamodule=data)
