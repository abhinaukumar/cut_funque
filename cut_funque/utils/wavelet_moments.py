import numpy as np
from ..utils.block_utils import im2block
from .custom_wavedec import custom_wavedec2
from .integral_image_utils import integral_image


def wavelet_moments_pyr(pyr_ref, pyr_dis=None, ret_x2=False, ret_y2=False):
    # Pyramids are assumed to have the structure
    # ([A1, ..., An], [(H1, V1, D1), ..., (Hn, Vn, Dn)])
    approxs_ref, details_ref = pyr_ref
    n_levels = len(approxs_ref)
    win_dim = (1 << n_levels)  # 2^L
    win_size = (1 << (n_levels << 1))  # 2^(2L), i.e., a win_dim X win_dim square

    mu_x = approxs_ref[-1] / win_dim
    var_x = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1) + details_ref[0][0].shape[2:])
    for detail_level_ref in details_ref:
        var_x_add = np.stack([np.abs(subband)**2 for subband in detail_level_ref], axis=-1).sum(-1)
        var_x = im2block(var_x, 2, 2).sum((0, 1)).reshape(var_x_add.shape) + var_x_add
    var_x /= win_size

    ret_tup = (mu_x, var_x)
    if ret_x2:
        s_x = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1) + details_ref[0][0].shape[2:])
        for detail_level_ref in details_ref:
            s_x_add = np.stack([subband**2 for subband in detail_level_ref], axis=-1).sum(-1)
            s_x = im2block(s_x, 2, 2).sum((0, 1)) + s_x_add
        s_x /= win_size
        ret_tup += (s_x,)

    if pyr_dis is not None:
        approxs_dis, details_dis = pyr_dis
        assert len(approxs_ref) == len(approxs_dis), 'Both wavelet pyramids must be of the same height'

        mu_y = approxs_dis[-1] / win_dim
        var_y = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1))
        cov_xy = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1))
        for detail_level_ref, detail_level_dis in zip(details_ref, details_dis):
            var_y_add = np.stack([np.abs(subband)**2 for subband in detail_level_dis], axis=-1).sum(-1)
            cov_xy_add = np.stack([subband_ref * np.conj(subband_dis) for subband_ref, subband_dis in zip(detail_level_ref, detail_level_dis)], axis=-1).sum(-1)
            var_y = im2block(var_y, 2, 2).sum((0, 1)) + var_y_add
            cov_xy = im2block(cov_xy, 2, 2).sum((0, 1)) + cov_xy_add

        var_y /= win_size
        cov_xy /= win_size
        ret_tup += (mu_y, var_y)
        if ret_y2:
            s_y = np.zeros((details_dis[0][0].shape[0] << 1, details_dis[0][0].shape[1] << 1) + details_dis[0][0].shape[2:])
            for detail_level_dis in details_dis:
                s_y_add = np.stack([subband**2 for subband in detail_level_dis], ayis=-1).sum(-1)
                s_y = im2block(s_y, 2, 2).sum((0, 1)) + s_y_add
            s_y /= win_size
            ret_tup += (s_y,)

        ret_tup += (cov_xy,)

    return ret_tup


def wavelet_multi_scale_moments_pyr(pyr_ref, pyr_dis=None, ret_x2=False, ret_y2=False):
    approxs_ref, details_ref = pyr_ref

    mu_xs = []
    var_xs = []

    var_x_cum = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1))
    win_dim = 1
    win_size = 1
    for approx_ref, detail_level_ref in zip(approxs_ref, details_ref):
        win_dim <<= 1
        win_size <<= 2

        var_x_add = np.stack([np.abs(subband)**2 for subband in detail_level_ref], axis=-1).sum(-1)
        var_x_cum = im2block(var_x_cum, 2, 2).sum((0, 1)) + var_x_add
        mu_x = approx_ref / win_dim
        var_x = var_x_cum / win_size

        mu_xs.append(mu_x)
        var_xs.append(var_x)

    ret_tup = (mu_xs, var_xs)
    if ret_x2:
        mu_x2s = []

        mu_x2_cum = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1) + details_ref[0][0].shape[2:])
        for approx_ref, detail_level_ref in zip(approxs_ref, details_ref):
            mu_x2_add = np.stack([subband**2 for subband in detail_level_ref], axis=-1).sum(-1)
            mu_x2_cum = im2block(mu_x2_cum, 2, 2).sum((0, 1)) + mu_x2_add
            mu_x2 = mu_x2_cum / win_size

            mu_x2s.append(mu_x2)
        ret_tup += (mu_x2s,)

    if pyr_dis is not None:
        mu_ys = []
        var_ys = []
        cov_xys = []
        approxs_dis, details_dis = pyr_dis

        assert len(approxs_ref) == len(approxs_dis), 'Both wavelet pyramids must be of the same height'

        var_y_cum = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1))
        cov_xy_cum = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1))

        win_dim = 1
        win_size = 1
        for approx_dis, detail_level_ref, detail_level_dis in zip(approxs_dis, details_ref, details_dis):
            win_dim <<= 1
            win_size <<= 2

            var_y_add = np.stack([np.abs(subband)**2 for subband in detail_level_dis], axis=-1).sum(-1)
            cov_xy_add = np.stack([subband_ref * np.conj(subband_dist) for subband_ref, subband_dist in zip(detail_level_ref, detail_level_dis)], axis=-1).sum(-1)

            var_y_cum = im2block(var_y_cum, 2, 2).sum((0, 1)) + var_y_add
            cov_xy_cum = im2block(cov_xy_cum, 2, 2).sum((0, 1)) + cov_xy_add

            mu_y = approx_dis / win_dim
            var_y = var_y_cum / win_size
            cov_xy = cov_xy_cum / win_size

            mu_ys.append(mu_y)
            var_ys.append(var_y)
            cov_xys.append(cov_xy)

        ret_tup += (mu_ys, var_ys)
        if ret_y2:
            mu_y2s = []

            mu_y2_cum = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1) + details_ref[0][0].shape[2:])
            for approx_ref, detail_level_ref in zip(approxs_ref, details_ref):
                mu_y2_add = np.stack([subband**2 for subband in detail_level_ref], axis=-1).sum(-1)
                mu_y2_cum = im2block(mu_y2_cum, 2, 2).sum((0, 1)) + mu_y2_add
                mu_y2 = mu_y2_cum / win_size

                mu_y2s.append(mu_y2)
            ret_tup += (mu_y2s,)
        ret_tup += (cov_xys,)

    return list(zip(*ret_tup))


def wavelet_moments(img_ref, img_dis=None, ret_x2=False, ret_y2=False, **kwargs):
    pyr_ref = custom_wavedec2(img_ref, **kwargs)
    if img_dis is not None:
        pyr_dis = custom_wavedec2(img_dis, **kwargs)
    else:
        pyr_dis = None
    return wavelet_moments_pyr(pyr_ref, pyr_dis, ret_x2, ret_y2)


def wavelet_multi_scale_moments(img_ref, img_dis=None, ret_x2=False, ret_y2=False, **kwargs):
    pyr_ref = custom_wavedec2(img_ref, **kwargs)
    if img_dis is not None:
        pyr_dis = custom_wavedec2(img_dis, **kwargs)
    else:
        pyr_dis = None
    return wavelet_multi_scale_moments_pyr(pyr_ref, pyr_dis, ret_x2, ret_y2)


def wavelet_moment_integral_images(pyr_ref, pyr_dis=None, ret_x2=False, ret_y2=False):
    # Pyramids are assumed to have the structure
    # ([A1, ..., An], [(H1, V1, D1), ..., (Hn, Vn, Dn)])
    approxs_ref, details_ref = pyr_ref
    n_levels = len(approxs_ref)
    mu_scale = (1 << n_levels)

    moment_map = approxs_ref[-1] * mu_scale  # Already divided by 2^L
    int_x = integral_image(moment_map)

    moment_map = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1) + details_ref[0][0].shape[2:])
    for detail_level_ref in details_ref:
        moment_map_add = np.stack([np.abs(subband)**2 for subband in detail_level_ref], axis=-1).sum(-1)
        moment_map = im2block(moment_map, 2, 2).sum((0, 1)) + moment_map_add
    moment_map = moment_map + np.abs(approxs_ref[-1])**2
    int_abs_x2 = integral_image(moment_map)

    ret_tup = (int_x, int_abs_x2)
    if ret_x2:
        moment_map = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1) + details_ref[0][0].shape[2:])
        for detail_level_ref in details_ref:
            moment_map_add = np.stack([subband**2 for subband in detail_level_ref], axis=-1).sum(-1)
            moment_map = im2block(moment_map, 2, 2).sum((0, 1)) + moment_map_add
        moment_map = moment_map + approxs_ref[-1]**2
        int_x2 = integral_image(moment_map)
        ret_tup += (int_x2,)

    if pyr_dis is not None:
        approxs_dis, details_dis = pyr_dis
        assert len(approxs_ref) == len(approxs_dis), 'Both wavelet pyramids must be of the same height'

        moment_map = approxs_dis[-1] * mu_scale  # Already divided by 2^L
        int_y = integral_image(moment_map)

        moment_map = np.zeros((details_dis[0][0].shape[0] << 1, details_dis[0][0].shape[1] << 1) + details_dis[0][0].shape[2:])
        for detail_level_dis in details_dis:
            moment_map_add = np.stack([np.abs(subband)**2 for subband in detail_level_dis], axis=-1).sum(-1)
            moment_map = im2block(moment_map, 2, 2).sum((0, 1)) + moment_map_add
        moment_map = moment_map + np.abs(approxs_dis[-1])**2
        int_abs_y2 = integral_image(moment_map)

        moment_map = np.zeros((details_ref[0][0].shape[0] << 1, details_ref[0][0].shape[1] << 1))
        for detail_level_ref, detail_level_dis in zip(details_ref, details_dis):
            moment_map_add = np.stack([subband_ref * np.conj(subband_dis) for subband_ref, subband_dis in zip(detail_level_ref, detail_level_dis)], axis=-1).sum(-1)
            moment_map = im2block(moment_map, 2, 2).sum((0, 1)) + moment_map_add
        moment_map = moment_map + approxs_ref[-1]*np.conj(approxs_dis[-1])
        int_xy = integral_image(moment_map)

        ret_tup += (int_y, int_abs_y2)
        if ret_y2:
            moment_map = np.zeros((details_dis[0][0].shape[0] << 1, details_dis[0][0].shape[1] << 1) + details_dis[0][0].shape[2:])
            for detail_level_dis in details_dis:
                moment_map_add = np.stack([subband**2 for subband in detail_level_dis], axis=-1).sum(-1)
                moment_map = im2block(moment_map, 2, 2).sum((0, 1)) + moment_map_add
            moment_map = moment_map + approxs_dis[-1]**2
            int_y2 = integral_image(moment_map)
            ret_tup += (int_y2,)

        ret_tup += (int_xy,)

    return ret_tup