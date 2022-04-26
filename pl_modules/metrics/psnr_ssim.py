from torchmetrics import PeakSignalNoiseRatio
from pl_modules.registry import PL_METRIC_REGISTRY
from pl_modules.metrics.metric_util import rgb2ycbcr


# Make consistent with BasicSR
class PSNR2(PeakSignalNoiseRatio):
    def __init__(self, crop_border, input_order='HWC', test_y_channel=False, *args, **kwargs):
        super(PSNR2, self).__init__(*args, **kwargs)
        assert input_order in ['HWC', 'CHW']
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_channel = test_y_channel

    def update(self, preds, target):
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

        return super().update(preds, target)



@PL_METRIC_REGISTRY.register()
def calculate_psnr(*args, **kwargs):
    return PSNR2(*args, **kwargs)

