from typing import Dict, Any, Optional
import numpy as np
import time

from videolib import Video
from videolib.buffer import CircularBuffer
from qualitylib.feature_extractor import FeatureExtractor
from qualitylib.result import Result

from ..utils.block_utils import multi_scale_block_means
from ..utils.custom_wavedec import custom_wavedec2, complex_add_pyrs
from ..utils.cut_utils import get_weights_from_stats, weighted_cut_sum
from ..utils.dnt import dnt
from ..utils.filter_utils import filter_pyr
from ..utils.lab_utils import get_lab, apply_hdrmax
from ..utils.resize_utils import imresize
from ..utils.wavelet_moments import wavelet_multi_scale_moments_pyr

from ..features.ssim_features import ssim_maps_from_moments, ssim_comp_maps_from_moments
from ..features.vif_features import vif_maps_from_moments, vif_comp_maps_from_moments
from ..features.rred_features import strred_maps_from_moments, strred_comp_maps_from_moments
from ..features.dlm_features import dlm_pyr_level
from ..features.nss_features import ggd_unif_cut_param_est, ggd_kld, ggd_param_est, aggd_unif_cut_param_est, aggd_kld, aggd_param_est


class CutFunqueFeatureExtractor(FeatureExtractor):
    NAME = 'Cut_FUNQUE_fex'
    VERSION = '1.0'
    feat_names = None
    def __init__(self, use_cache: bool = True, sample_rate: Optional[int] = None) -> None:
        super().__init__(use_cache, sample_rate)
        self._num_bins = 4
        self._cut_types = ['lum', 'spat', 'temp']
        self._space = 'pucolor'
        self._mscn_types = ['l', 'l_sig', 'c', 'c_sig']
        self._prod_dirs = ['h', 'v', 'd1', 'd2']

        self._buf_size = 4
        self._lab_ref_buf = CircularBuffer(self._buf_size)
        self._lab_dis_buf = CircularBuffer(2)
        self._lab_hdrmax_ref_buf = CircularBuffer(2)
        self._lab_hdrmax_dis_buf = CircularBuffer(2)

        self._ssim_keys = ['ssim_l_mu', 'ssim_l_sig', 'ssim_c_mu', 'ssim_c_sig']
        self._vif_keys = ['vif_l', 'tvif_l', 'vif_c', 'tvif_c']
        self._strred_keys = ['srred_l', 'trred_l', 'srred_c', 'trred_c']
        self._dlm_keys = ['dlm_l']
        self._ggd_keys = [f'ggd{param}_{mscn_type}_mscn' for mscn_type in self._mscn_types for param in ['scale', 'shape']]
        self._kld_keys = [f'kld_{mscn_type}_mscn' for mscn_type in self._mscn_types]
        self._aggd_keys = [f'aggd{param}_prod_{mscn_type}_mscn' for mscn_type in self._mscn_types for param in ['scale_l', 'scale_r', 'mu', 'shape']]
        self._kld_prod_keys = [f'kld_prod_{mscn_type}_mscn' for mscn_type in self._mscn_types]
        self._qual_keys = self._ssim_keys + self._vif_keys + self._strred_keys + self._dlm_keys + self._kld_keys + self._kld_prod_keys

        # Indices of quality and distortion features
        self._qual_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 12], dtype='int')
        self._dist_inds = np.array([8, 9, 10, 11] + list(range(13, len(self._qual_keys))), dtype='int')

        self._skip_levels = 2  # Ensures smallest window is of size 8
        self._cut_scales = 4
        self._wavelet_levels = self._skip_levels + self._cut_scales
        self._wavelet = 'haar'
        self._csf = 'watson'
        self.feat_names = self._get_feat_names()

    def _get_feat_names(self):
        feat_names = []
        for space in [self._space, f'{self._space}_hdrmax']:
            space_feat_names = []
            # Glob NSS features
            for qual_key in self._ggd_keys:
                space_feat_names.append(f'{space}_glob_{qual_key}')
            for qual_key in self._kld_keys:
                space_feat_names.append(f'{space}_glob_{qual_key}')
            for qual_key in self._aggd_keys:
                space_feat_names.append(f'{space}_glob_{qual_key}')
            for qual_key in self._kld_prod_keys:
                space_feat_names.append(f'{space}_glob_{qual_key}')
            # Unweighted aggregated features
            for qual_key in self._qual_keys:
                space_feat_names.append(f'{space}_{qual_key}_unweighted')
            # Binned-weighted aggregated features
            for cut_type in self._cut_types:
                for qual_key in self._qual_keys:
                    space_feat_names.append(f'{space}_{qual_key}_{cut_type}')

            feat_names.extend(space_feat_names)

        return feat_names

    def _get_cut_weights(self, moments):
        self._l_buf_sum_arr = self._l_buf_sum_arr - self._lab_ref_buf.front()[..., 0] + self._lab_ref_buf.back()[..., 0]
        self._l_buf_sum2_arr = self._l_buf_sum2_arr - self._lab_ref_buf.front()[..., 0]**2 + self._lab_ref_buf.back()[..., 0]**2

        temp_mu = self._l_buf_sum_arr / self._buf_size
        temp_mu2 = self._l_buf_sum2_arr / self._buf_size
        temp_std = np.sqrt(np.abs(temp_mu2 - temp_mu**2))
        temp_covs = []
        temp_mus = multi_scale_block_means(temp_mu, self._skip_levels, self._cut_scales)
        temp_stds = multi_scale_block_means(temp_std, self._skip_levels, self._cut_scales)
        temp_covs = [temp_std / np.clip(temp_mu, 1e-6, None) for temp_mu, temp_std in zip(temp_mus, temp_stds)]
        del temp_mu, temp_mu2, temp_mus, temp_stds

        cut_weights = []
        for moment_scale, temp_cov_scale in zip(moments, temp_covs):
            spat_mu_scale = moment_scale[0]
            spat_var_scale = moment_scale[1]
            spat_cov_scale = np.sqrt(spat_var_scale) / np.clip(spat_mu_scale, 1e-6, None)
            cut_weights.append(np.stack([get_weights_from_stats(cut_feats.flatten(), self._num_bins) for cut_feats in [spat_mu_scale, spat_cov_scale, temp_cov_scale]], 0))
        return cut_weights

    def _combine_bins(self, bin_feats, bin_axis):
        qual_feats = bin_feats[..., self._qual_inds]
        dist_feats = bin_feats[..., self._dist_inds]
        qual_bin_agg_feats = np.min(qual_feats, axis=bin_axis)  # Worst quality = min quality
        dist_bin_agg_feats = np.max(dist_feats, axis=bin_axis)  # Worst quality = max distortion
        bin_agg_feats = np.zeros(qual_bin_agg_feats.shape[:-1] + (bin_feats.shape[-1],))
        bin_agg_feats[..., self._qual_inds] = qual_bin_agg_feats
        bin_agg_feats[..., self._dist_inds] = dist_bin_agg_feats
        return bin_agg_feats

    def _combine_scales(self, scale_feats, scale_axis):
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363])
        weights = weights / np.sum(weights)
        n_scales = scale_feats.shape[scale_axis]
        pre_broadcast_shape = np.ones((scale_feats.ndim,))
        pre_broadcast_shape[scale_axis] = n_scales
        scale_agg_feats = 0
        for s, weight in enumerate(weights[:n_scales]):
            scale_agg_feats = scale_agg_feats + weight * np.take(scale_feats, s, axis=scale_axis)
        return scale_agg_feats

    def _extract_features(self, lab_ref, lab_dis, lab_ref_prev, lab_dis_prev, cut_weights=None):
        mscn_l_ref, (_, sig_l_ref) = dnt(lab_ref[..., 0], full=True, noise_sig=4e-3)
        mscn_l_dis, (_, sig_l_dis) = dnt(lab_dis[..., 0], full=True, noise_sig=4e-3)
        mscn_l_sig_ref = dnt(sig_l_ref, noise_sig=4e-3)
        mscn_l_sig_dis = dnt(sig_l_dis, noise_sig=4e-3)
        del sig_l_ref, sig_l_dis

        mscn_l_ref_moments_ms = wavelet_multi_scale_moments_pyr(custom_wavedec2(mscn_l_ref, 'haar', levels=self._wavelet_levels))[self._skip_levels:]
        mscn_l_sig_ref_moments_ms = wavelet_multi_scale_moments_pyr(custom_wavedec2(mscn_l_sig_ref, 'haar', levels=self._wavelet_levels))[self._skip_levels:]
        mscn_l_dis_moments_ms = wavelet_multi_scale_moments_pyr(custom_wavedec2(mscn_l_dis, 'haar', levels=self._wavelet_levels))[self._skip_levels:]
        mscn_l_sig_dis_moments_ms = wavelet_multi_scale_moments_pyr(custom_wavedec2(mscn_l_sig_dis, 'haar', levels=self._wavelet_levels))[self._skip_levels:]

        mscn_c_ref, (_, sig_c_ref) = dnt(np.linalg.norm(lab_ref[..., 1:], axis=-1), full=True, noise_sig=4e-3)
        mscn_c_dis, (_, sig_c_dis) = dnt(np.linalg.norm(lab_dis[..., 1:], axis=-1), full=True, noise_sig=4e-3)
        mscn_c_sig_ref = dnt(sig_c_ref, noise_sig=4e-3)
        mscn_c_sig_dis = dnt(sig_c_dis, noise_sig=4e-3)
        del sig_c_ref, sig_c_dis

        mscn_c_ref_moments_ms = wavelet_multi_scale_moments_pyr(custom_wavedec2(mscn_c_ref, 'haar', levels=self._wavelet_levels))[self._skip_levels:]
        mscn_c_sig_ref_moments_ms = wavelet_multi_scale_moments_pyr(custom_wavedec2(mscn_c_sig_ref, 'haar', levels=self._wavelet_levels))[self._skip_levels:]
        mscn_c_dis_moments_ms = wavelet_multi_scale_moments_pyr(custom_wavedec2(mscn_c_dis, 'haar', levels=self._wavelet_levels))[self._skip_levels:]
        mscn_c_sig_dis_moments_ms = wavelet_multi_scale_moments_pyr(custom_wavedec2(mscn_c_sig_dis, 'haar', levels=self._wavelet_levels))[self._skip_levels:]

        mscns_ref = [mscn_l_ref, mscn_l_sig_ref, mscn_c_ref, mscn_c_sig_ref]
        mscns_dis = [mscn_l_dis, mscn_l_sig_dis, mscn_c_dis, mscn_c_sig_dis]
        mscn_momentss_ref = [mscn_l_ref_moments_ms, mscn_l_sig_ref_moments_ms, mscn_c_ref_moments_ms, mscn_c_sig_ref_moments_ms]
        mscn_momentss_dis = [mscn_l_dis_moments_ms, mscn_l_sig_dis_moments_ms, mscn_c_dis_moments_ms, mscn_c_sig_dis_moments_ms]

        mscn_prods_ref = []
        for mscn in [mscn_l_ref, mscn_l_sig_ref, mscn_c_ref, mscn_c_sig_ref]:
            mscn_prods_ref.append([])
            mscn_prods_ref[-1].append(mscn*np.roll(mscn, 1, axis=0))
            mscn_prods_ref[-1].append(mscn*np.roll(mscn, 1, axis=1))
            mscn_prods_ref[-1].append(mscn*np.roll(np.roll(mscn, 1, axis=0), 1, axis=1))
            mscn_prods_ref[-1].append(mscn*np.roll(np.roll(mscn, 1, axis=0), -1, axis=1))
        mscn_prods_dis = []
        for mscn in [mscn_l_dis, mscn_l_sig_dis, mscn_c_dis, mscn_c_sig_dis]:
            mscn_prods_dis.append([])
            mscn_prods_dis[-1].append(mscn*np.roll(mscn, 1, axis=0))
            mscn_prods_dis[-1].append(mscn*np.roll(mscn, 1, axis=1))
            mscn_prods_dis[-1].append(mscn*np.roll(np.roll(mscn, 1, axis=0), 1, axis=1))
            mscn_prods_dis[-1].append(mscn*np.roll(np.roll(mscn, 1, axis=0), -1, axis=1))

        mu_neg2_mscn_prodss_ref = [[multi_scale_block_means(np.clip(mscn_prods, None, 0)**2, self._skip_levels, self._cut_scales) for mscn_prods in mscn_prods_ref_type] for mscn_prods_ref_type in mscn_prods_ref]
        mu_negcount_mscn_prodss_ref = [[multi_scale_block_means((mscn_prods < 0).astype('float64'), self._skip_levels, self._cut_scales) for mscn_prods in mscn_prods_ref_type] for mscn_prods_ref_type in mscn_prods_ref]
        mu_abs2_mscn_prodss_ref = [[multi_scale_block_means(np.abs(mscn_prods)**2, self._skip_levels, self._cut_scales) for mscn_prods in mscn_prods_ref_type] for mscn_prods_ref_type in mscn_prods_ref]
        mu_abs_mscn_prodss_ref = [[multi_scale_block_means(np.abs(mscn_prods), self._skip_levels, self._cut_scales) for mscn_prods in mscn_prods_ref_type] for mscn_prods_ref_type in mscn_prods_ref]

        mu_neg2_mscn_prodss_dis = [[multi_scale_block_means(np.clip(mscn_prods, None, 0)**2, self._skip_levels, self._cut_scales) for mscn_prods in mscn_prods_dis_type] for mscn_prods_dis_type in mscn_prods_dis]
        mu_negcount_mscn_prodss_dis = [[multi_scale_block_means((mscn_prods < 0).astype('float64'), self._skip_levels, self._cut_scales) for mscn_prods in mscn_prods_dis_type] for mscn_prods_dis_type in mscn_prods_dis]
        mu_abs2_mscn_prodss_dis = [[multi_scale_block_means(np.abs(mscn_prods)**2, self._skip_levels, self._cut_scales) for mscn_prods in mscn_prods_dis_type] for mscn_prods_dis_type in mscn_prods_dis]
        mu_abs_mscn_prodss_dis = [[multi_scale_block_means(np.abs(mscn_prods), self._skip_levels, self._cut_scales) for mscn_prods in mscn_prods_dis_type] for mscn_prods_dis_type in mscn_prods_dis]

        pyrs_ref = [filter_pyr(custom_wavedec2(lab_ref[..., channel_ind], wavelet=self._wavelet, levels=self._wavelet_levels), self._csf, channel=channel_ind) for channel_ind in range(3)]
        pyrs_dis = [filter_pyr(custom_wavedec2(lab_dis[..., channel_ind], wavelet=self._wavelet, levels=self._wavelet_levels), self._csf, channel=channel_ind) for channel_ind in range(3)]

        pyr_l_ref = pyrs_ref[0]
        pyr_l_dis = pyrs_dis[0]

        pyr_c_ref = complex_add_pyrs(*pyrs_ref[1:])
        pyr_c_dis = complex_add_pyrs(*pyrs_dis[1:])
        del pyrs_ref, pyrs_dis

        dlm_nums, dlm_dens = tuple(zip(*[dlm_pyr_level(pyr_l_ref[1][lev], pyr_l_dis[1][lev]) for lev in range(self._skip_levels, self._wavelet_levels)]))

        l_moments_ms = wavelet_multi_scale_moments_pyr(pyr_l_ref, pyr_l_dis)[self._skip_levels:]
        c_moments_ms = wavelet_multi_scale_moments_pyr(pyr_c_ref, pyr_c_dis, ret_x2=True, ret_y2=True)[self._skip_levels:]
        del pyr_l_ref, pyr_l_dis, pyr_c_ref, pyr_c_dis

        # Use wavelet moments to compute cut weights if they haven't been provided
        if cut_weights is None:
            cut_weights = self._get_cut_weights(l_moments_ms)

        pyrs_ref = [filter_pyr(custom_wavedec2(lab_ref[..., channel_ind] - lab_ref_prev[..., channel_ind], wavelet=self._wavelet, levels=self._wavelet_levels), self._csf, channel=channel_ind) for channel_ind in range(3)]
        pyrs_dis = [filter_pyr(custom_wavedec2(lab_dis[..., channel_ind] - lab_dis_prev[..., channel_ind], wavelet=self._wavelet, levels=self._wavelet_levels), self._csf, channel=channel_ind) for channel_ind in range(3)]

        pyr_l_ref = pyrs_ref[0]
        pyr_l_dis = pyrs_dis[0]

        pyr_c_ref = complex_add_pyrs(*pyrs_ref[1:])
        pyr_c_dis = complex_add_pyrs(*pyrs_dis[1:])

        del pyrs_ref, pyrs_dis

        l_diff_moments_ms = wavelet_multi_scale_moments_pyr(pyr_l_ref, pyr_l_dis)[self._skip_levels:]
        c_diff_moments_ms = wavelet_multi_scale_moments_pyr(pyr_c_ref, pyr_c_dis, ret_x2=True, ret_y2=True)[self._skip_levels:]
        del pyr_l_ref, pyr_l_dis, pyr_c_ref, pyr_c_dis

        def _process_scale(scale, cut_weight_scale):
            unweighted_feats = []
            weighted_feats = []
            l_moments = l_moments_ms[scale]
            c_moments = c_moments_ms[scale]
            l_diff_moments = l_diff_moments_ms[scale]
            c_diff_moments = c_diff_moments_ms[scale]

            # SSIM
            ssim_l_maps = list(map(lambda x: (1-x)**3, ssim_maps_from_moments(l_moments)))
            ssim_c_maps = list(map(lambda x: (1-x)**3, ssim_comp_maps_from_moments(c_moments[:2] + c_moments[3:5] + c_moments[-1:])))
            unweighted_feats.extend(1 - np.mean(ssim_map)**(1/3) for ssim_map in ssim_l_maps + ssim_c_maps)
            weighted_feats.extend(1 - weighted_cut_sum(ssim_map.reshape(-1, 1), cut_weight_scale)**(1/3) for ssim_map in ssim_l_maps + ssim_c_maps)

            # VIF
            vif_l_num_map, vif_l_den_map = vif_maps_from_moments(l_moments)
            unweighted_feats.append(np.mean(vif_l_num_map/vif_l_den_map))
            weighted_feats.append(weighted_cut_sum(vif_l_num_map.reshape(-1, 1) / vif_l_den_map.reshape(-1, 1), cut_weight_scale))
            tvif_l_num_map, tvif_l_den_map = vif_maps_from_moments(l_diff_moments)
            unweighted_feats.append(np.mean(tvif_l_num_map/tvif_l_den_map))
            weighted_feats.append(weighted_cut_sum(tvif_l_num_map.reshape(-1, 1) / tvif_l_den_map.reshape(-1, 1), cut_weight_scale))
            vif_c_num_map, vif_c_den_map = vif_comp_maps_from_moments(c_moments)
            unweighted_feats.append(np.mean(vif_c_num_map/vif_c_den_map))
            weighted_feats.append(weighted_cut_sum(vif_c_num_map.reshape(-1, 1) / vif_c_den_map.reshape(-1, 1), cut_weight_scale))
            tvif_c_num_map, tvif_c_den_map = vif_comp_maps_from_moments(c_diff_moments)
            unweighted_feats.append(np.mean(tvif_c_num_map/tvif_c_den_map))
            weighted_feats.append(weighted_cut_sum(tvif_c_num_map.reshape(-1, 1) / tvif_c_den_map.reshape(-1, 1), cut_weight_scale))

            # ST-RRED
            srred_l_map, trred_l_map = strred_maps_from_moments(l_moments, l_diff_moments)  # B x 2
            unweighted_feats.append(np.mean(srred_l_map))
            unweighted_feats.append(np.mean(trred_l_map))
            weighted_feats.append(weighted_cut_sum(srred_l_map.reshape(-1, 1), cut_weight_scale))
            weighted_feats.append(weighted_cut_sum(trred_l_map.reshape(-1, 1), cut_weight_scale))
            srred_c_map, trred_c_map = strred_comp_maps_from_moments(c_moments, c_diff_moments)  # B x 2
            unweighted_feats.append(np.mean(srred_c_map))
            unweighted_feats.append(np.mean(trred_c_map))
            weighted_feats.append(weighted_cut_sum(srred_c_map.reshape(-1, 1), cut_weight_scale))
            weighted_feats.append(weighted_cut_sum(trred_c_map.reshape(-1, 1), cut_weight_scale))

            # DLM
            dlm_nums_maps = dlm_nums[scale]
            dlm_dens_maps = dlm_dens[scale]
            dlm_map = (sum(dlm_nums_maps)**(1/3) + 1e-4) / (sum(dlm_dens_maps)**(1/3) + 1e-4)
            unweighted_feats.append(np.mean(dlm_map))
            weighted_feats.append(weighted_cut_sum(dlm_map.reshape(-1, 1), cut_weight_scale))

            # NSS
            mscn_args_pack = mscns_ref, mscns_dis, mscn_momentss_ref, mscn_momentss_dis
            # GGD KLDs
            for coeff_ref, coeff_dis, mscn_moments_ref, mscn_moments_dis in zip(*mscn_args_pack):
                mscn_ref_params = ggd_unif_cut_param_est(coeff_ref, mscn_moments_ref, scale, lambda x: 2**(self._skip_levels+x+1))
                mscn_dis_params = ggd_unif_cut_param_est(coeff_dis, mscn_moments_dis, scale, lambda x: 2**(self._skip_levels+x+1))
                kld_map = ggd_kld(mscn_ref_params, mscn_dis_params)
                unweighted_feats.append(np.mean(kld_map))
                weighted_feats.append(weighted_cut_sum(kld_map.reshape(-1, 1), cut_weight_scale))

            # NSS (Prod)
            mscn_prod_ref_arg_pack = mu_neg2_mscn_prodss_ref, mu_negcount_mscn_prodss_ref, mu_abs2_mscn_prodss_ref, mu_abs_mscn_prodss_ref
            mscn_prod_dis_arg_pack = mu_neg2_mscn_prodss_dis, mu_negcount_mscn_prodss_dis, mu_abs2_mscn_prodss_dis, mu_abs_mscn_prodss_dis
            # AGGD Params

            for mscn_prod_ref_type_args, mscn_prod_dis_type_args in zip(zip(*mscn_prod_ref_arg_pack), zip(*mscn_prod_dis_arg_pack)):
                kld_prod_map = 0
                for mscn_prod_ref_args, mscn_prod_dis_args in zip(zip(*mscn_prod_ref_type_args), zip(*mscn_prod_dis_type_args)):
                    mscn_prod_ref_params = aggd_unif_cut_param_est(*mscn_prod_ref_args, scale)
                    mscn_prod_dis_params = aggd_unif_cut_param_est(*mscn_prod_dis_args, scale)
                    kld_prod_map = kld_prod_map + aggd_kld(mscn_prod_ref_params, mscn_prod_dis_params)
                kld_prod_map = kld_prod_map / len(self._prod_dirs)
                unweighted_feats.append(np.mean(kld_prod_map))
                weighted_feats.append(weighted_cut_sum(kld_prod_map.reshape(-1, 1), cut_weight_scale))

            unweighted_feats = np.array(unweighted_feats)
            weighted_feats = np.concatenate(weighted_feats, -1)
            return unweighted_feats, weighted_feats

        feats = []

        # Global NSS
        glob_feats = []
        mscn_ref_params = []
        for mscn in [mscn_l_ref, mscn_l_sig_ref, mscn_c_ref, mscn_c_sig_ref]:
            mscn_ref_params.append(np.array(ggd_param_est(mscn)))
        mscn_dis_params = []
        for mscn in [mscn_l_dis, mscn_l_sig_dis, mscn_c_dis, mscn_c_sig_dis]:
            mscn_dis_params.append(np.array(ggd_param_est(mscn)))
        # GGD Params
        glob_feats.extend(mscn_dis_params)
        # KLD
        glob_feats.append(np.array([ggd_kld(param_ref, param_dis) for param_ref, param_dis in zip(mscn_ref_params, mscn_dis_params)]))

        # Global NSS (Prod)
        mscn_prod_ref_params = []
        for mscn_prod_type in mscn_prods_ref:
            mscn_prod_ref_params_type = []
            for mscn_prod in mscn_prod_type:
                mscn_prod_ref_params_type.append(np.array(aggd_param_est(mscn_prod)))
            mscn_prod_ref_params.append(np.stack(mscn_prod_ref_params_type, 0))  # 4 x F
        mscn_prod_dis_params = []
        for mscn_prod_type in mscn_prods_dis:
            mscn_prod_dis_params_type = []
            for mscn_prod in mscn_prod_type:
                mscn_prod_dis_params_type.append(np.array(aggd_param_est(mscn_prod)))
            mscn_prod_dis_params.append(np.stack(mscn_prod_dis_params_type, 0))  # #dirs x F
        # AGGD Params
        glob_feats.extend(np.mean(np.stack(mscn_prod_dis_params, 0), axis=1))
        # KLD (Prod)
        prod_klds = np.array([np.mean([aggd_kld(param_ref, param_dis) for param_ref, param_dis in zip(mscn_prod_ref_params_type, mscn_prod_dis_params_type)]) for mscn_prod_ref_params_type, mscn_prod_dis_params_type in zip(mscn_prod_ref_params, mscn_prod_dis_params)]) 
        glob_feats.append(prod_klds)

        feats.extend(glob_feats)

        unweighted_feats = []
        weighted_feats = []
        for scale, cut_weight_scale in enumerate(cut_weights):
            unweighted_agg_feats, weighted_agg_feats = _process_scale(scale, cut_weight_scale)
            unweighted_feats.append(unweighted_agg_feats)
            weighted_feats.append(weighted_agg_feats)

        unweighted_feats = np.stack(unweighted_feats, axis=0) # scales x feats
        weighted_feats = np.stack(weighted_feats, axis=0)  # scales x cut types x bins x feats
        weighted_feats = np.transpose(weighted_feats, (1, 0, 2, 3))  # cut types x scales x bins x feats

        feats.append(self._combine_scales(unweighted_feats, scale_axis=0).flatten())
        weighted_bin_comb_feats = self._combine_bins(weighted_feats, bin_axis=2)
        feats.append(self._combine_scales(weighted_bin_comb_feats, scale_axis=1).flatten())
        feats = np.concatenate(feats, 0)
        return feats, cut_weights  # Useful when cut_weights computed here must be reused.

    def _add_frames_to_bufs(self, frame_ref, frame_dis):
        # Add reference frame to buffer
        self._lab_ref_buf.check_append(get_lab(frame_ref, self._space))
        # Add distorted frame to buffer
        self._lab_dis_buf.check_append(get_lab(frame_dis, self._space))
        # Add HDRMAX-ed reference frame to buffer
        self._lab_hdrmax_ref_buf.check_append(apply_hdrmax(self._lab_ref_buf[0]))
        # Add HDRMAX-ed distorted frame to buffer
        self._lab_hdrmax_dis_buf.check_append(apply_hdrmax(self._lab_dis_buf[0]))

    def _run_on_frame(self, frame_ref, frame_dis):
        # Update buffers
        self._add_frames_to_bufs(frame_ref, frame_dis)
        # Get current and previous frames from buffers
        lab_ref = self._lab_ref_buf[0]
        lab_ref_prev = self._lab_ref_buf[1]
        lab_dis = self._lab_dis_buf[0]
        lab_dis_prev = self._lab_dis_buf[1]
        lab_hdrmax_ref = self._lab_hdrmax_ref_buf[0]
        lab_hdrmax_ref_prev = self._lab_hdrmax_ref_buf[1]
        lab_hdrmax_dis = self._lab_hdrmax_dis_buf[0]
        lab_hdrmax_dis_prev = self._lab_hdrmax_dis_buf[1]

        # Compute reference wavelet pyramid to derive cut weights
        frame_feats = []
        # Compute features without HDRMAX
        feats, cut_weights = self._extract_features(lab_ref, lab_dis, lab_ref_prev, lab_dis_prev, cut_weights=None)
        frame_feats.append(feats)

        # Compute features with HDRMAX
        feats, _ = self._extract_features(lab_hdrmax_ref, lab_hdrmax_dis, lab_hdrmax_ref_prev, lab_hdrmax_dis_prev, cut_weights=cut_weights)
        frame_feats.append(feats)

        frame_feats = np.concatenate(frame_feats, 0)
        return frame_feats

    def _run_on_asset(self, asset_dict: Dict[str, Any]) -> Result:
        proc_width = asset_dict['width'] // 2
        proc_height = asset_dict['height'] // 2
        factor = 2**self._wavelet_levels
        h_pad = ((proc_height + factor-1) // factor) * factor
        w_pad = ((proc_width + factor-1) // factor) * factor

        self._l_buf_sum_arr = np.zeros((h_pad, w_pad))
        self._l_buf_sum2_arr = np.zeros((h_pad, w_pad))

        with Video(
            asset_dict['ref_path'], mode='r',
            standard=asset_dict['ref_standard'],
            width=asset_dict['width'], height=asset_dict['height']
            ) as v_ref:
            with Video(
                asset_dict['dis_path'], mode='r',
                standard=asset_dict['dis_standard'],
                width=asset_dict['width'], height=asset_dict['height']
                ) as v_dis:
                # feats = np.zeros((1, len(self.feat_names)))  # Use if only temporal average is to be returned
                feats = []  # Use if framewise features are to be returned
                start = time.time()
                for frame_ind, (frame_ref, frame_dis) in enumerate(zip(v_ref, v_dis)):
                    frame_ref.yuv = imresize(frame_ref.yuv, output_shape=(proc_height, proc_width))
                    frame_dis.yuv = imresize(frame_dis.yuv, output_shape=(proc_height, proc_width))

                    frame_ref.yuv = np.pad(frame_ref.yuv, [(0, h_pad-proc_height), (0, w_pad-proc_width), (0, 0)], mode='reflect')
                    frame_dis.yuv = np.pad(frame_dis.yuv, [(0, h_pad-proc_height), (0, w_pad-proc_width), (0, 0)], mode='reflect')

                    frame_feats = self._run_on_frame(frame_ref, frame_dis)
                    if np.any(np.isnan(frame_feats)) or np.any(np.isinf(frame_feats)):
                        raise ValueError('Encountered NaN!')
                    # feats = feats + np.expand_dims(frame_feats, 0)  # Use if only temporal average is to be returned
                    feats.append(np.expand_dims(frame_feats, 0))  # Use if framewise features are to be returned

                    print(f'Processed frame {frame_ind}/{v_ref.num_frames} of {asset_dict["asset_id"]} after {time.time() - start} s')

        # feats = feats / v_ref.num_frames  # Use if only temporal average is to be returned
        feats = np.concatenate(feats, 0)  # Use if framewise features are to be returned
        print(f'Processed {asset_dict["dis_path"]}')
        return self._to_result(asset_dict, feats, self.feat_names)
