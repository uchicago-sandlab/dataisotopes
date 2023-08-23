from PIL import Image
import numpy as np
import torch
from io import BytesIO


def jpeg_compression(im, quality=75):
    """ JPEG compression taken from https://github.com/facebookarchive/adversarial_image_defenses/blob/master/adversarial/lib/defenses.py"""
    savepath = BytesIO()
    im.save(savepath, 'JPEG', quality=quality)
    im = Image.open(savepath).convert('RGB')
    return im