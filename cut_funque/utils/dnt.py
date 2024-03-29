import scipy.ndimage as nd
import numpy as np

def dnt(x, k=5, full=False, eps=4e-3, noise_sig=None):
    '''
    Divisive normalization transform
    '''
    if noise_sig:
        x = x + np.random.randn(*x.shape)*noise_sig
    mu = nd.uniform_filter(x, k)
    sig_sq = np.abs(nd.uniform_filter(x**2, k) - mu**2)
    # Saturation constant tuned to an input range of 0-1 (+ filtering)
    sig = np.sqrt(sig_sq)
    mscns = (x - mu) / (sig + eps)
    if full:
        return mscns, (mu, sig)
    else:
        return mscns
