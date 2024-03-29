import numpy as np

def im2block(img, k, stride=1):
    # Parameters
    m, n = img.shape[:2]
    s0, s1 = img.strides[:2]
    nrows = m - k + 1
    ncols = n - k + 1
    shape = (k, k, nrows, ncols) + img.shape[2:]
    arr_stride = (s0, s1, s0, s1) + img.strides[2:]

    ret = np.lib.stride_tricks.as_strided(img, shape=shape, strides=arr_stride)
    return ret[:, :, ::stride, ::stride]

def im2col(img, k, stride=1):
    return im2block(img, k, stride).reshape(k*k, -1, *img.shape[2:])

def multi_scale_block_means(img, start_level, num_levels):
    win_size = 2**(start_level+1)
    img_means = []
    img = np.mean(im2block(img, win_size, win_size), axis=(0, 1))
    img_means.append(img)
    for lev in range(num_levels-1):
        img = np.mean(im2block(img, 2, 2), axis=(0, 1))
        img_means.append(img)
    return img_means