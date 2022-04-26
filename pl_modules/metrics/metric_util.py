import torch
from torchmetrics import Metric

def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.
    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    if y_only:
        out_img = torch.matmul(img, torch.tensor([65.481, 128.553, 24.966], device=img.device)) + 16.0
    else:
        out_img = torch.matmul(
            img, torch.tensor([[65.481, -37.797, 112.0],
                               [128.553, -74.203, -93.786],
                               [24.966, 112.0, -18.214]],
                              device=img.device)) + torch.tensor([16, 128, 128], device=img.device)
    out_img /= 255.
    return out_img


class GatherImages(Metric):
    def __init__(self):
        super(GatherImages, self).__init__(compute_on_step=False)

        self.add_state('lr', default=[], dist_reduce_fx=None)
        self.add_state('lr_hr', default=[], dist_reduce_fx=None)
        self.add_state('hr', default=[], dist_reduce_fx=None)

    def update(self, lr, lr_hr, hr):
        self.lr.append(lr.squeeze(0).cpu())
        self.lr_hr.append(lr_hr.squeeze(0).cpu())
        self.hr.append(hr.squeeze(0).cpu())

    def compute(self):
        return self.lr, self.lr_hr, self.hr