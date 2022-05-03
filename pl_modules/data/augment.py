import random
import string
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def Upper_Lower_string(length):  # define the function and pass the length as argument
    # Print the string in Lowercase
    result = ''.join(
        (random.choice(string.ascii_letters + string.digits + string.punctuation) for x in range(length)))  # run loop until the define length
    return result


def augment_add(imgs, font_paths, text=True):
    if not isinstance(imgs, list):
        imgs = [imgs]

    if text and random.random() < 0.1:
        font_path = random.choice(font_paths)
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

        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            img = Image.fromarray((img*255).astype(np.uint8))
            scale = h / max_h
            size_c = int(size * scale)
            font = ImageFont.truetype(font_path, size_c)
            coords_c = (coords * scale).astype(int)
            img_editable = ImageDraw.Draw(img)
            img_editable.fontmode = "PA"
            img_editable.text(tuple(coords_c), text, tuple(color), font=font)
            imgs[i] = np.array(img) / 255.
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs
