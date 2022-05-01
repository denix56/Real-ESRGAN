import torchmetrics
from pl_modules.registry import PL_METRIC_REGISTRY
from pl_modules.metrics.metric_util import rgb2ycbcr


class BaseMetric:
    def __init__(self, crop_border:int, input_order:str='HWC', test_y_channel:bool=False):
        assert input_order in ['HWC', 'CHW']
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_channel = test_y_channel

    def prepare(self, preds, target):
        if self.input_order == 'HWC':
            preds = preds.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        if self.crop_border != 0:
            crop_border = self.crop_border
            preds = preds[..., crop_border:-crop_border, crop_border:-crop_border]
            target = target[..., crop_border:-crop_border, crop_border:-crop_border]

        if self.test_y_channel:
            preds = rgb2ycbcr(preds.permute(0, 2, 3, 1), True)
            target = rgb2ycbcr(target.permute(0, 2, 3, 1), True)
        return preds, target


# Make consistent with BasicSR
class PSNR(BaseMetric, torchmetrics.PeakSignalNoiseRatio):
    def __init__(self, crop_border:int, input_order:str='HWC', test_y_channel:bool=False, *args, **kwargs):
        super(BaseMetric, self).__init__(crop_border, input_order, test_y_channel)
        super(torchmetrics.PeakSignalNoiseRatio, self).__init__(*args, **kwargs)

    def update(self, preds, target):
        preds, target = self.prepare(preds, target)
        return super().update(preds, target)


# Make consistent with BasicSR
class SSIM(BaseMetric, torchmetrics.StructuralSimilarityIndexMeasure):
    def __init__(self, crop_border:int, input_order:str='HWC', test_y_channel:bool=False, *args, **kwargs):
        super(BaseMetric, self).__init__(crop_border, input_order, test_y_channel)
        super(torchmetrics.StructuralSimilarityIndexMeasure, self).__init__(*args, **kwargs)

    def update(self, preds, target):
        preds, target = self.prepare(preds, target)
        return super().update(preds, target)


# Make consistent with BasicSR
class LPIPS(BaseMetric, torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity):
    def __init__(self, crop_border:int, input_order:str='HWC', test_y_channel:bool=False, *args, **kwargs):
        super(BaseMetric, self).__init__(crop_border, input_order, test_y_channel)
        super(torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity, self).__init__(*args, **kwargs)

    def update(self, preds, target):
        preds, target = self.prepare(preds, target)
        return super().update(preds, target)


# Make consistent with BasicSR
class FID(BaseMetric, torchmetrics.FrechetInceptionDistance):
    def __init__(self, crop_border:int, input_order:str='HWC', test_y_channel:bool=False, *args, **kwargs):
        super(BaseMetric, self).__init__(crop_border, input_order, test_y_channel)
        super(torchmetrics.FrechetInceptionDistance, self).__init__(*args, **kwargs)

    def update(self, preds, target):
        preds, target = self.prepare(preds, target)
        super().update(preds, real=False)
        super().update(target, real=True)



@PL_METRIC_REGISTRY.register()
def calculate_psnr(*args, **kwargs):
    return PSNR(*args, **kwargs)


@PL_METRIC_REGISTRY.register()
def calculate_ssim(*args, **kwargs):
    return SSIM(*args, **kwargs)


@PL_METRIC_REGISTRY.register()
def calculate_lpips(*args, **kwargs):
    return LPIPS(*args, **kwargs)


@PL_METRIC_REGISTRY.register()
def calculate_fid(*args, **kwargs):
    return FID(*args, **kwargs)

