from copy import deepcopy

from pl_modules import PL_METRIC_REGISTRY
from pl_modules.metrics.psnr_ssim import calculate_psnr
# from .niqe import calculate_niqe
# from .psnr_ssim import calculate_psnr, calculate_ssim

__all__ = ['calculate_psnr']


def build_metric(opt):
    """Build metric from data and options.
        Args:
            opt (dict): Configuration. It must contain:
                type (str): Model type.
        """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = PL_METRIC_REGISTRY.get(metric_type)(**opt)
    return metric