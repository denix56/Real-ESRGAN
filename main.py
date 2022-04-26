# flake8: noqa
import os
import os.path as osp
from pl_modules.train import train_pipeline


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, os.pardir))
    train_pipeline(root_path)