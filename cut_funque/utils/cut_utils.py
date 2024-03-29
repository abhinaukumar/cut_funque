import numpy as np


def get_weights_from_stats(stats, num_bins=4):
    minval, maxval = np.min(stats), np.max(stats)
    sig = np.clip((maxval - minval) / (2*(num_bins-1)), 1e-6, None)
    mus = np.linspace(minval, maxval, num_bins)
    mus = np.expand_dims(mus, axis=tuple(range(stats.ndim)))
    weight_mat = np.exp(-0.5*(stats[..., None] - mus)**2/sig**2)
    return weight_mat

def weighted_cut_sum(vals, weights):
    '''
    vals: ... x C x F
    weights: ... x C x B
    return: ... x B x F
    '''
    if vals.ndim < 2 or weights.ndim < 2:
        raise ValueError('vals and weights must both have at least two dims')
    if vals.ndim > weights.ndim:
        _vals = vals
        _weights = np.expand_dims(weights, tuple(range(vals.ndim - weights.ndim)))
    elif weights.ndim > vals.ndim:
        _vals = np.expand_dims(vals, tuple(range(weights.ndim - vals.ndim)))
        _weights = weights
    else:
        _vals = vals
        _weights = weights

    _vals = np.expand_dims(_vals, -2)
    _weights = np.expand_dims(_weights, -1)

    _weight_norms = np.sum(_weights, -3)  # Sum over all cuts -> ... x B x 1
    return np.sum(_vals * _weights, -3) / _weight_norms  # ... x B x F
