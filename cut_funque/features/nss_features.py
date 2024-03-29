import numpy as np
from scipy.special import gamma
from ..utils.ggd_utils import GGD, AGGD
from ..utils.block_utils import im2block

_ggd = GGD()
_aggd = AGGD()


def ggd_kld(param_refs, param_diss):
    b_refs, alpha_refs = param_refs
    b_diss, alpha_diss = param_diss
    klds = ((b_refs / b_diss)**alpha_diss * gamma((alpha_diss + 1) / alpha_refs) - gamma((alpha_refs + 1)/alpha_refs)) / gamma(1 / alpha_refs) + \
            np.log(alpha_refs / (2 * b_refs * gamma(1 / alpha_refs))) - np.log(alpha_diss / (2 * b_diss * gamma(1 / alpha_diss)))
    return klds

def ggd_param_est(x):
    mu_x = np.mean(x)
    std_x = np.std(x)
    mu_abs = np.mean(np.abs(x - mu_x))
    gam_hat = np.where(std_x != 0, mu_abs**2 / std_x**2, np.ones_like(std_x))
    alpha_ind = _ggd.inverse_lookup(gam_hat)
    alpha = _ggd.alphas[alpha_ind]
    b = std_x / np.sqrt(gamma(3/alpha)/gamma(1/alpha))
    return b, alpha


def ggd_unif_cut_param_est(x, x_moments_ms, scale, win_size_func):
    win_size = win_size_func(scale)
    mu_x, var_x = x_moments_ms[scale]
    std_x = np.sqrt(var_x)
    mu_abs = np.mean(np.abs(im2block(x, win_size, win_size) - np.expand_dims(mu_x, axis=(0, 1))), axis=(0, 1))
    gam_hat = np.where(std_x != 0, mu_abs**2 / std_x**2, np.ones_like(std_x))
    alpha_ind = _ggd.inverse_lookup(gam_hat)
    alpha = _ggd.alphas[alpha_ind]
    b = std_x / np.sqrt(gamma(3/alpha)/gamma(1/alpha))
    return b, alpha


def aggd_kld(param_refs, param_diss):
    b_l_refs, b_r_refs, _, alpha_refs = param_refs
    b_l_diss, b_r_diss, _, alpha_diss = param_diss
    klds = (gamma((alpha_diss + 1)/alpha_refs)*(b_l_refs**(alpha_diss+1)/b_l_diss**alpha_diss + b_r_refs**(alpha_diss+1)/b_r_diss**alpha_diss)/(b_l_refs + b_r_refs) - gamma((alpha_refs+1)/alpha_refs)) / gamma(1/alpha_refs) + \
           np.log(alpha_refs/((b_l_refs + b_r_refs)*gamma(1/alpha_refs))) - np.log(alpha_diss/((b_l_diss + b_r_diss)*gamma(1/alpha_diss)))
    return klds


def aggd_param_est(x):
    sig_l = np.mean(x[x < 0]**2)
    if np.isnan(sig_l):
        sig_l = 1e-6
    sig_r = np.mean(x[x >= 0]**2)
    if np.isnan(sig_r):
        sig_r = 1e-6
    mu_abs = np.mean(np.abs(x))
    mu_2 = np.mean(x**2)
    ratio = np.sqrt(sig_r/sig_l)
    gam_hat = mu_abs**2 / mu_2
    gam_hat = gam_hat * (ratio**3+1) * (ratio+1) / (ratio**2+1)**1
    alpha_ind = _aggd.inverse_lookup(gam_hat)
    alpha = _aggd.alphas[alpha_ind]
    b_l = np.sqrt(sig_l * gamma(3/alpha) / gamma(1/alpha))
    b_r = np.sqrt(sig_r * gamma(3/alpha) / gamma(1/alpha))
    mu = (b_l - b_r) * gamma(2/alpha) / gamma(1/alpha)
    return b_l, b_r, mu, alpha


def aggd_unif_cut_param_est(mu_neg_x2_scales, mu_negcount_scales, mu_x2_scales, mu_abs_scales, scale):
    sig_l = np.clip(mu_neg_x2_scales[scale] * mu_negcount_scales[scale], 1e-6, None)
    mu_2 = mu_x2_scales[scale]
    sig_r = np.clip((mu_2 - sig_l*mu_negcount_scales[scale]) / np.clip(1 - mu_negcount_scales[scale], 1e-6, 1), 1e-6, None)
    mu_abs = mu_abs_scales[scale]
    ratio = np.sqrt(sig_r/sig_l)
    gam_hat = np.clip(mu_abs**2 / mu_2, 1e-6, None)
    gam_hat = gam_hat * (ratio**3+1) * (ratio+1) / (ratio**2+1)**1
    alpha_ind = _aggd.inverse_lookup(gam_hat)
    alpha = _aggd.alphas[alpha_ind]
    b_l = np.sqrt(sig_l * gamma(3/alpha) / gamma(1/alpha))
    b_r = np.sqrt(sig_r * gamma(3/alpha) / gamma(1/alpha))
    mu = (b_l - b_r) * gamma(2/alpha) / gamma(1/alpha)
    return b_l, b_r, mu, alpha
