"""
A set of functions used to downsample and upsample image for the PNAS paper.

It uses the idea that human eye is more sensitive to the value variation than to hue variation. As inverting the Lippmann spectrum is costly, to speed up the computation one can calculate the specturm (and hence hue) on a sparse grid, and combine it whith orignal (dense) value.
"""

import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage.draw import disk
from skimage.transform import resize


def sample_image(image, n_samples, dot_radius_mm, ppmm=None, width_mm=None):

    h, w, _ = image.shape
    mask = np.full((h, w), 0, dtype=np.uint8)
    result = np.zeros((n_samples, n_samples, 3), dtype=np.uint8)

    if ppmm is None:
        ppmm = w/width_mm
    else:
        width_mm = w/ppmm

    height_mm = h/ppmm

    for x in range(n_samples):
        for y in range(n_samples):
            Ycoords, Xcoords = disk(((y + 0.5) * w/(n_samples), (x + 0.5) * w/(n_samples)), dot_radius_mm*ppmm)
            Ycoords = np.clip(Ycoords, a_min=0, a_max=h-1)
            Xcoords = np.clip(Xcoords, a_min=0, a_max=w-1)
            mask[Ycoords, Xcoords] = 1
            result[y, x] = np.uint8(np.mean(image[Ycoords, Xcoords], axis=0))

    return result, mask, (width_mm, height_mm)

def clever_upsample(original, subsampled, order):
    small_hsv = rgb2hsv(subsampled)
    small_hsv[:, :, 2] = 255
    large_hsv = rgb2hsv(resize(hsv2rgb(small_hsv), original.shape, order=order))
    large_hsv[:, :, 2] = rgb2hsv(original)[:, :, 2]
    return  hsv2rgb(large_hsv)


def set_ticks_in_mm(axes, idx, image, im_size_mm, n_ticks):
    axes[idx].set_xticks(np.linspace(-0.5, image.shape[1] - 0.5, n_ticks))
    axes[idx].set_xticklabels(np.around(np.linspace(0, im_size_mm[0], n_ticks), 0))
    axes[idx].set_yticks(np.linspace(-0.5, image.shape[0] - 0.5, n_ticks))
    axes[idx].set_yticklabels(np.around(np.linspace(0, im_size_mm[1], n_ticks)[::-1], 0))
    axes[idx].set_xlabel("mm")
    axes[idx].set_ylabel("mm")
