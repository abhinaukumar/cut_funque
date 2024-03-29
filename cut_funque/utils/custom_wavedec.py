from pywt import dwt2


def custom_wavedec2(data, wavelet, mode='symmetric', levels=None, axes=(-2, -1)):
    approxs = []
    details = []
    if levels is None:
        levels = 1
    for _ in range(levels):
        wavelet_level = dwt2(data, wavelet, mode, axes)
        approxs.append(wavelet_level[0])
        details.append(wavelet_level[1])
        data = wavelet_level[0]
    return (approxs, details)


def pyr_diff(pyr_1, pyr_2):
    approxs_1, details_1 = pyr_1
    approxs_2, details_2 = pyr_2

    approxs_diff = [approx_1 - approx_2 for approx_1, approx_2 in zip(approxs_1, approxs_2)]
    details_diff = [tuple(map(lambda details: details[0] - details[1], zip(detail_lev, detail_lev_prev))) for detail_lev, detail_lev_prev in zip(details_1, details_2)]
    return approxs_diff, details_diff


def complex_add_pyrs(pyr_real, pyr_imag):
    approxs_real, details_real = pyr_real
    approxs_imag, details_imag = pyr_imag
    approxs_comp = list(map(lambda subs: subs[0] + 1j*subs[1], zip(approxs_real, approxs_imag)))
    details_comp = list(tuple(map(lambda subs: subs[0] + 1j*subs[1], zip(details_lev_real, details_lev_imag))) for details_lev_real, details_lev_imag in zip(details_real, details_imag))
    return approxs_comp, details_comp
