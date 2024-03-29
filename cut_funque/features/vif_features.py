import numpy as np
from ..utils.wavelet_moments import wavelet_moments_pyr, wavelet_multi_scale_moments_pyr


def vif_maps_from_moments(moments):
    sigma_nsq = 0.1
    _, var_x, _, var_y, cov_xy = moments

    g = cov_xy / (var_x + 1e-10)
    sv_sq = np.abs(var_y - g * cov_xy)
    nums = np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4
    dens = np.log(1 + var_x / sigma_nsq) + 1e-4

    return nums, dens

def vif_comp_maps_from_moments(moments):
    sigma_nsq = 0.2
    _, var_x, s_x, _, var_y, _, cov_xy = moments

    var_x_real = (var_x + np.real(s_x))/2
    var_x_imag = (var_x - np.real(s_x))/2
    cov_x = np.imag(s_x)/2

    g = np.conj(cov_xy) / (var_x + 1e-10)
    sv_sq = np.abs(var_y - np.abs(g * cov_xy))

    g_s_gT_mat_00 = np.real(g)**2*var_x_real + np.imag(g)**2*var_x_imag - 2*np.real(g)*np.imag(g)*cov_x
    g_s_gT_mat_11 = np.real(g)**2*var_x_imag + np.imag(g)**2*var_x_real + 2*np.real(g)*np.imag(g)*cov_x
    g_s_gT_mat_01 = cov_x*(np.real(g)**2 - np.imag(g)**2) + np.real(g)*np.imag(g)*(var_x_real - var_x_imag)

    nums = np.abs(np.log(np.abs((g_s_gT_mat_00 + 0.5*(sv_sq + sigma_nsq))*(g_s_gT_mat_11 + 0.5*(sv_sq + sigma_nsq)) - g_s_gT_mat_01**2)) - 2*np.log(0.5*(sv_sq + sigma_nsq))) + 1e-4
    dens = np.abs(np.log(np.abs((var_x_real + 0.5*sigma_nsq)*(var_x_imag + 0.5*sigma_nsq) - cov_x**2)) - 2*np.log(0.5*sigma_nsq)) + 1e-4

    return nums, dens

def ms_vif_maps_from_moments(ms_moments):
    return list(map(vif_maps_from_moments, ms_moments))

def ms_vif_comp_maps_from_moments(ms_moments):
    return list(map(vif_comp_maps_from_moments, ms_moments))

def vif_maps_pyr(pyr_ref, pyr_dis):
    return vif_maps_from_moments(wavelet_moments_pyr(pyr_ref, pyr_dis))

def vif_comp_maps_pyr(pyr_ref, pyr_dis):
    return vif_comp_maps_from_moments(wavelet_moments_pyr(pyr_ref, pyr_dis, ret_x2=True))

def ms_vif_maps_pyr(pyr_ref, pyr_dis):
    return ms_vif_maps_from_moments(wavelet_multi_scale_moments_pyr(pyr_ref, pyr_dis))

def ms_vif_comp_maps_pyr(pyr_ref, pyr_dis):
    return ms_vif_comp_maps_from_moments(wavelet_multi_scale_moments_pyr(pyr_ref, pyr_dis, ret_x2=True))
