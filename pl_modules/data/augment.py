import math
import random
import string
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import torch
import torch.nn as nn

import kornia as K
import math


def Upper_Lower_string(length):  # define the function and pass the length as argument
    # Print the string in Lowercase
    result = ''.join(
        (random.choice(string.ascii_letters + string.digits + string.punctuation) for x in range(length)))  # run loop until the define length
    return result


def augment_add(imgs, font_paths, text=True, c_shuffle=False):
    if not isinstance(imgs, list):
        imgs = [imgs]

    if text and random.random() < 0.1:
        size = random.randint(8, 32)
        text = Upper_Lower_string(random.randint(5, 20))
        color = np.random.randint(0, 256, size=3)
        max_h, max_w = 0, 0
        for img in imgs:
            h, w = img.shape[:2]
            if h > max_h:
                max_h, max_w = h, w
        coords = np.array([random.randint(5, max_h - size - 5),
                           random.randint(5, max_w - size - 5)])

        while True:
            font_path = random.choice(font_paths)
            fonts = []
            scales = []
            try:
                for i, img in enumerate(imgs):
                    h, w = img.shape[:2]
                    scale = h / max_h
                    size_c = int(size * scale)
                    font = ImageFont.truetype(font_path, size_c)
                    fonts.append(font)
                    scales.append(scale)

                imgs_new = []
                for i, (font, scale, img) in enumerate(zip(fonts, scales, imgs)):
                    img = Image.fromarray((img * 255).astype(np.uint8))
                    coords_c = (coords * scale).astype(int)
                    img_editable = ImageDraw.Draw(img)
                    img_editable.fontmode = "PA"
                    img_editable.text(tuple(coords_c), text, tuple(color), font=font)
                    imgs_new.append(np.array(img) / 255.)
                imgs = imgs_new
                break
            except OSError:
                pass
                
    if c_shuffle and random.random() < 0.25:
        for i, img in enumerate(imgs):
            c_list = np.arange(img.shape[2])
            np.random.default_rng().shuffle(c_list)
            imgs[i] = np.ascontiguousarray(img[..., c_list])
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs


def rgb_to_oklab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to OkLAB.

    .. image:: _static/img/rgb_to_oklab.png

    Args:
        image: RGB Image to be converted to OkLAB with shape :math:`(*, 3, H, W)`.

    Returns:
         OkLAB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_oklab(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image = K.color.rgb_to_linear_rgb(image)

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    l_: torch.Tensor = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m_: torch.Tensor = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s_: torch.Tensor = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = torch.pow(l_, 1./3)
    m_ = torch.pow(m_, 1./3)
    s_ = torch.pow(s_, 1./3)

    l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    out: torch.Tensor = torch.stack([l, a, b], -3)

    return out


def oklab_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a OkLAB image to RGB.

    .. image:: _static/img/oklab_to_rgb.png

    Args:
        image: OkLAB Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
         RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_oklab(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    l: torch.Tensor = image[..., 0, :, :]
    a: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    l_: torch.Tensor = l + 0.3963377774 * a + 0.2158037573 * b
    m_: torch.Tensor = l - 0.1055613458 * a - 0.0638541728 * b
    s_: torch.Tensor = l - 0.0894841775 * a - 1.2914855480 * b

    l_ = l_*l_*l_
    m_ = m_*m_*m_
    s_ = s_*s_*s_

    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    b = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    out: torch.Tensor = torch.stack([r, g, b], -3)
    out = K.color.linear_rgb_to_rgb(out)

    return out


class RgbToOklab(nn.Module):
    r"""Convert an image from RGB to OkLAB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        OkLAB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> oklab = RgbToOklab()
        >>> output = oklab(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_oklab(image)


class OklabToRgb(nn.Module):
    r"""Convert an image from OkLAB to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = OklabToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return oklab_to_rgb(image)


def rgb_to_hsi(image: torch.Tensor, eps: float) -> torch.Tensor:
    r"""Convert an image from RGB to HSI.

    .. image:: _static/img/rgb_to_hsi.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSI version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and I are in the range 0..1.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsi(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :].clamp(0, 1)
    g: torch.Tensor = image[..., 1, :, :].clamp(0, 1)
    b: torch.Tensor = image[..., 2, :, :].clamp(0, 1)

    mask = g < b
    h: torch.Tensor = torch.acos((0.5 * ((r-g)+(r-b)))/(torch.sqrt((r-g)**2 + (r-b)*(g-b)) + eps))
    h[mask] = 2*math.pi - h[mask]

    i: torch.Tensor = (r + g + b) / 3
    s: torch.Tensor = 1 - image.min(dim=-3)[0] / i
    s[i < eps] = 0

    return torch.stack((h, s, i), dim=-3)


def hsi_to_rgb(image: torch.Tensor, eps: float) -> torch.Tensor:
    r"""Convert an image from HSI to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and I are in the range 0..1.

    Args:
        image: HSI Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsi_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :].clamp(0, 2*math.pi)
    s: torch.Tensor = image[..., 1, :, :].clamp(0, 1)
    i: torch.Tensor = image[..., 2, :, :].clamp(0, 1)

    h[h >= 2*math.pi] -= 2*math.pi

    h_ = h.clone()
    h_[h_ > 4*math.pi/3] -= 4*math.pi/3
    h_[h_ > 2*math.pi/3] -= 2*math.pi/3

    c = torch.cos(h_) / (torch.cos(math.pi/3 - h_) + eps)

    r: torch.Tensor = c.clone()
    g: torch.Tensor = c.clone()
    b: torch.Tensor = c.clone()

    mask = h < 2*math.pi/3
    g[mask] = 1 - g[mask]
    b[mask] = -1

    mask = (h >= 2*math.pi/3) & (h <= 4*math.pi/3)
    r[mask] = -1
    b[mask] = 1 - b[mask]

    mask = h > 4*math.pi/3
    r[mask] = 1 - r[mask]
    g[mask] = -1

    i.unsqueeze_(-3)
    s.unsqueeze_(-3)
    out = torch.stack((r, g, b), dim=-3)*i*s + i

    return out


class RgbToHsi(nn.Module):
    r"""Convert an image from RGB to HSI.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        HSI version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hsi = RgbToHsi()
        >>> output = hsi(input)  # 2x3x4x5
    """

    def __init__(self, eps:float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_hsi(image, self.eps)


class HsiToRgb(nn.Module):
    r"""Convert an image from HSI to RGB.

    H channel values are assumed to be in the range 0..2pi. S and I are in the range 0..1.

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HsiToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self, eps:float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return hsi_to_rgb(image, self.eps)
