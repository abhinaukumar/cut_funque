import numpy as np
import scipy as sp

from .transfer_utils import get_transfer_mat_lms
from .pu_utils import pucolor


def get_lab(frame, space='pucolor'):
    transfer_mat_lms = get_transfer_mat_lms()
    lab = None
    if space == 'linear_yuv':
        return frame.linear_yuv
    if space == 'yuv':
        return frame.yuv / frame.standard.range
    if space == 'pucolor':
        lab = pucolor(frame.standard.linear_range * frame.xyz @ transfer_mat_lms.T)[..., np.array([0, 2, 1])]  # Order similar to YCrCb
    return lab


def apply_hdrmax(img, size=17, scale=4):
    if img.ndim == 2:
        mins = sp.ndimage.minimum_filter(img, size=size)
        maxs = sp.ndimage.maximum_filter(img, size=size)
        dens = np.clip(maxs - mins, 1e-6, None)
        img_scaled = 2 * (img - mins) / dens - 1
        img_exp = np.sign(img_scaled) * (np.exp(scale*np.abs(img_scaled)) - 1)
        return img_exp
    else:
        if img.shape[-1] == 3:
            l_hdrmax = apply_hdrmax(img[..., 0], size, scale)
            c_hdrmax = apply_hdrmax(np.linalg.norm(img[..., 1:], axis=-1), size, scale)
            arg = np.arctan2(img[..., 2], img[..., 1])
            img_exp = np.stack([l_hdrmax, c_hdrmax*np.cos(arg), c_hdrmax*np.sin(arg)], axis=-1)
            return img_exp
        else:
            return np.stack([apply_hdrmax(img[..., i]) for i in range(img.shape[-1])], axis=-1)
