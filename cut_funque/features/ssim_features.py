import numpy as np
from ..utils.wavelet_moments import wavelet_moments_pyr, wavelet_multi_scale_moments_pyr

_C1 = 0.01
_C2 = 0.03


def ssim_maps_from_moments(moments):
    mu_x, var_x, mu_y, var_y, cov_xy = moments
    l_map = (2*mu_x*mu_y + _C1) / (np.abs(mu_x)**2 + np.abs(mu_y)**2 + _C1)
    cs_map = (2*cov_xy + _C2) / (var_x + var_y + _C2)
    return l_map, cs_map

def ssim_comp_maps_from_moments(moments):
    mu_x, var_x, mu_y, var_y, cov_xy = moments
    l_map = (2*np.abs(mu_x*mu_y) + _C1) / (np.abs(mu_x)**2 + np.abs(mu_y)**2 + _C1)
    cs_map = (2*np.abs(cov_xy) + _C2) / (var_x + var_y + _C2)
    return np.clip(l_map, 0, 1), np.clip(cs_map, 0, 1)

def ms_ssim_maps_from_moments(ms_moments):
    return list(map(ssim_maps_from_moments, ms_moments))

def ms_ssim_maps_pyr(pyr_ref, pyr_dis):
    return ms_ssim_maps_from_moments(wavelet_multi_scale_moments_pyr(pyr_ref, pyr_dis))

def ssim_maps_pyr(pyr_ref, pyr_dis):
    return ssim_maps_from_moments(wavelet_moments_pyr(pyr_ref, pyr_dis))
