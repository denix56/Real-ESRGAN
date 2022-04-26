from copy import deepcopy
from pl_modules.registry import PL_MODEL_REGISTRY

import importlib
from basicsr.utils import scandir
from os import path as osp

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'pl_modules.models.{file_name}') for file_name in model_filenames]


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = PL_MODEL_REGISTRY.get(opt['model_type'])(opt)
    print(f'Model [{model.__class__.__name__}] is created.')

    return model