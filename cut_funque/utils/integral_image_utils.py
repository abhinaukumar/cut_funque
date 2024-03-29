from typing import Tuple, Union
import numpy as np


def integral_image(img: np.ndarray) -> np.ndarray:
    '''
    Compute integral image aka summed area table.

    Args:
        img: Input image.

    Returns:
        np.ndarray: Integral image.
    '''
    rows, cols = img.shape[:2]
    int_img = np.zeros((rows+1, cols+1) + img.shape[2:], dtype=img.dtype)
    # int_img[1:, 1:] = np.cumsum(np.cumsum(img, 0), 1)
    # Slower, but more accurate than cumsum
    for i in range(rows):
        int_img[i+1, 1:] = int_img[i, 1:] + img[i, :]
    for j in range(cols):
        int_img[:, j+1] = int_img[:, j+1] + int_img[:, j]
    return int_img


def integral_image_sum(int_img: np.ndarray, inds: Union[Tuple[int, int], np.ndarray], size: Union[Tuple[int, int], np.ndarray]) -> float:
    '''
    Obtain sum over a window from an integral image.

    Args:
        int_img: Integral image.
        inds: Indices of the top-left corner of the window (0-based)
        size: Size along each dimenson of the window.

    Returns:
        float: Sum over the given window.
    '''
    if isinstance(inds, (Tuple, list)):
        i, j = inds
    elif isinstance(inds, np.ndarray):
        i = inds[..., 0]
        j = inds[..., 1]

    if isinstance(size, Tuple):
        rows, cols = size
    elif isinstance(size, np.ndarray):
        rows = size[..., 0]
        cols = size[..., 1]

    return int_img[i, j] + int_img[i+rows, j+cols] - int_img[i+rows, j] - int_img[i, j+cols]


def moment_integral_images(img1, img2=None, ret_x2=False, ret_y2=False):
    int_x = integral_image(img1)
    int_abs_x2 = integral_image(np.abs(img1)**2)
    ret_tup = (int_x, int_abs_x2)

    if ret_x2:
        int_x2 = integral_image(img1**2)
        ret_tup += (int_x2,)

    if img2 is not None:
        int_y = integral_image(img2)
        int_abs_y2 = integral_image(np.abs(img2)**2)
        int_xy = integral_image(img1*np.conj(img2))
        ret_tup += (int_y, int_abs_y2)

        if ret_y2:
            int_y2 = integral_image(img2**2)
            ret_tup += (int_y2,)

        ret_tup += (int_xy,)

    return ret_tup


def integral_image_moments(moment_ints, win_size, stride=1):
    x_only = len(moment_ints) <= 3
    ret_x2 = False
    ret_y2 = False
    if len(moment_ints) == 2:
        int_x, int_abs_x2 = moment_ints
    elif len(moment_ints) == 3:
        int_x, int_abs_x2, int_x2 = moment_ints
        ret_x2 = True
    elif len(moment_ints) == 5:
        int_x, int_abs_x2, int_y, int_abs_y2, int_xy = moment_ints
    elif len(moment_ints) == 6:
        int_x, int_abs_x2, int_x2, int_y, int_abs_y2, int_xy = moment_ints
        ret_x2 = True
    elif len(moment_ints) == 7:
        int_x, int_abs_x2, int_x2, int_y, int_abs_y2, int_y2, int_xy = moment_ints
        ret_x2 = True
        ret_y2 = True

    norm = win_size * win_size
    mu_x = (int_x[:-win_size:stride, :-win_size:stride] - int_x[:-win_size:stride, win_size::stride] - int_x[win_size::stride, :-win_size:stride] + int_x[win_size::stride, win_size::stride])/norm
    mu_abs_x2 = (int_abs_x2[:-win_size:stride, :-win_size:stride] - int_abs_x2[:-win_size:stride, win_size::stride] - int_abs_x2[win_size::stride, :-win_size:stride] + int_abs_x2[win_size::stride, win_size::stride])/norm
    var_x = np.abs(mu_abs_x2 - np.abs(mu_x)**2)

    ret_tup = (mu_x, var_x)
    if ret_x2:
        mu_x2 = (int_x2[:-win_size:stride, :-win_size:stride] - int_x2[:-win_size:stride, win_size::stride] - int_x2[win_size::stride, :-win_size:stride] + int_x2[win_size::stride, win_size::stride])/norm
        s_x = mu_x2 - mu_x**2
        ret_tup += (s_x,)

    if not x_only:
        mu_y = (int_y[:-win_size:stride, :-win_size:stride] - int_y[:-win_size:stride, win_size::stride] - int_y[win_size::stride, :-win_size:stride] + int_y[win_size::stride, win_size::stride])/norm
        mu_abs_y2 = (int_abs_y2[:-win_size:stride, :-win_size:stride] - int_abs_y2[:-win_size:stride, win_size::stride] - int_abs_y2[win_size::stride, :-win_size:stride] + int_abs_y2[win_size::stride, win_size::stride])/norm
        var_y = mu_abs_y2 - np.abs(mu_y)**2
        ret_tup += (mu_y, var_y)
        if ret_y2:
            mu_y2 = (int_y2[:-win_size:stride, :-win_size:stride] - int_y2[:-win_size:stride, win_size::stride] - int_y2[win_size::stride, :-win_size:stride] + int_y2[win_size::stride, win_size::stride])/norm
            s_y = mu_y2 - mu_y**2
            ret_tup += (s_y,)
        mu_xy = (int_xy[:-win_size:stride, :-win_size:stride] - int_xy[:-win_size:stride, win_size::stride] - int_xy[win_size::stride, :-win_size:stride] + int_xy[win_size::stride, win_size::stride])/norm
        cov_xy = mu_xy - mu_x*mu_y

        ret_tup += (cov_xy,)

    return ret_tup
