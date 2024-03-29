import numpy as np
from ..utils.wavelet_moments import wavelet_moments_pyr, wavelet_multi_scale_moments_pyr

_entr_const = 0.5*np.log(2*np.pi*np.exp(1))
def entr_scale_maps_from_moments(moments):
    sigma_nsq = 0.1
    _, var_x = moments

    entrs = 0.5*np.log(var_x + sigma_nsq) + _entr_const
    scales = np.log(1 + var_x)
    return entrs, scales

def entr_scale_comp_maps_from_moments(moments):
    sigma_nsq = 0.2
    _, var_x, s_x = moments

    var_x_real = (var_x + np.real(s_x))/2
    var_x_imag = (var_x - np.real(s_x))/2
    cov_x = np.imag(s_x)/2

    entrs = np.log(np.abs((var_x_real + 0.5*sigma_nsq)*(var_x_imag + 0.5*sigma_nsq) - cov_x**2)) + 2*_entr_const
    scales = np.log(1 + var_x)
    return entrs, scales 

def ms_entr_scale_maps_from_moments(ms_moments):
    return list(map(entr_scale_maps_from_moments, ms_moments))

def ms_entr_scale_comp_maps_from_moments(ms_moments):
    return list(map(entr_scale_comp_maps_from_moments, ms_moments))

def entr_scale_maps_pyr(pyr_ref, pyr_dis):
    return entr_scale_maps_from_moments(wavelet_moments_pyr(pyr_ref, pyr_dis))

def entr_scale_comp_maps_pyr(pyr_ref, pyr_dis):
    return entr_scale_comp_maps_from_moments(wavelet_moments_pyr(pyr_ref, pyr_dis, ret_x2=True))

def ms_entr_scale_maps_pyr(pyr_ref, pyr_dis):
    return ms_entr_scale_comp_maps_from_moments(wavelet_multi_scale_moments_pyr(pyr_ref, pyr_dis))

def ms_entr_scale_comp_maps_pyr(pyr_ref, pyr_dis):
    return ms_entr_scale_comp_maps_from_moments(wavelet_multi_scale_moments_pyr(pyr_ref, pyr_dis, ret_x2=True))

def strred_maps_from_moments(moments, moments_diff=None):
    moments_ref = moments[:2]
    moments_dis = moments[2:4]
    spat_entr_ref, spat_scale_ref = entr_scale_maps_from_moments(moments_ref)
    spat_entr_dis, spat_scale_dis = entr_scale_maps_from_moments(moments_dis)
    srred_map = np.abs(spat_entr_ref * spat_scale_ref - spat_entr_dis * spat_scale_dis)

    if moments_diff is not None:
        moments_ref_diff = moments_diff[:2]
        moments_dis_diff = moments_diff[2:4]
        temp_entr_ref, temp_scale_ref = entr_scale_maps_from_moments(moments_ref_diff)
        temp_entr_dis, temp_scale_dis = entr_scale_maps_from_moments(moments_dis_diff)

        trred_map = np.abs(temp_entr_ref * temp_scale_ref * spat_scale_ref - temp_entr_dis * temp_scale_dis * spat_scale_dis)
        strred_maps = (srred_map, trred_map)
    else:
        strred_maps = (srred_map, np.zeros_like(srred_map))

    return strred_maps

def strred_comp_maps_from_moments(moments, moments_diff=None):
    moments_ref = moments[:3]
    moments_dis = moments[3:6]
    spat_entr_ref, spat_scale_ref = entr_scale_comp_maps_from_moments(moments_ref)
    spat_entr_dis, spat_scale_dis = entr_scale_comp_maps_from_moments(moments_dis)
    srred_map = np.abs(spat_entr_ref * spat_scale_ref - spat_entr_dis * spat_scale_dis)

    if moments_diff is not None:
        moments_ref_diff = moments_diff[:3]
        moments_dis_diff = moments_diff[3:6]
        temp_entr_ref, temp_scale_ref = entr_scale_comp_maps_from_moments(moments_ref_diff)
        temp_entr_dis, temp_scale_dis = entr_scale_comp_maps_from_moments(moments_dis_diff)

        trred_map = np.abs(temp_entr_ref * temp_scale_ref * spat_scale_ref - temp_entr_dis * temp_scale_dis * spat_scale_dis)
        strred_maps = (srred_map, trred_map)
    else:
        strred_maps = (srred_map, np.zeros_like(srred_map))

    return strred_maps

def ms_strred_maps_from_moments(ms_moments, ms_moments_diff=None):
    if ms_moments_diff is None:
        ms_moments_diff = [None]*len(ms_moments)
    return [strred_maps_from_moments(moments, moments_diff) for moments, moments_diff in zip(ms_moments, ms_moments_diff)]

def ms_strred_comp_maps_from_moments(ms_moments, ms_moments_diff=None):
    if ms_moments_diff is None:
        ms_moments_diff = [None]*len(ms_moments)
    return [strred_comp_maps_from_moments(moments, moments_diff) for moments, moments_diff in zip(ms_moments, ms_moments_diff)]

def strred_maps_pyr(pyr_ref, pyr_dis, pyr_ref_diff=None, pyr_dis_diff=None):
    return strred_maps_from_moments(*list(map(lambda pyr: wavelet_moments_pyr(pyr) if pyr is not None else None, [pyr_ref, pyr_dis, pyr_ref_diff, pyr_dis_diff])))

def strred_comp_maps_pyr(pyr_ref, pyr_dis, pyr_ref_diff=None, pyr_dis_diff=None):
    return strred_comp_maps_from_moments(*list(map(lambda pyr: wavelet_moments_pyr(pyr, ret_x2=True) if pyr is not None else None, [pyr_ref, pyr_dis, pyr_ref_diff, pyr_dis_diff])))

def ms_strred_maps_pyr(pyr_ref, pyr_dis, pyr_ref_diff=None, pyr_dis_diff=None):
    return ms_strred_maps_from_moments(*list(map(lambda pyr: wavelet_multi_scale_moments_pyr(pyr) if pyr is not None else None, [pyr_ref, pyr_dis, pyr_ref_diff, pyr_dis_diff])))

def ms_strred_comp_maps_pyr(pyr_ref, pyr_dis, pyr_ref_diff=None, pyr_dis_diff=None):
    return ms_strred_comp_maps_from_moments(*list(map(lambda pyr: wavelet_multi_scale_moments_pyr(pyr, ret_x2=True) if pyr is not None else None, [pyr_ref, pyr_dis, pyr_ref_diff, pyr_dis_diff])))
