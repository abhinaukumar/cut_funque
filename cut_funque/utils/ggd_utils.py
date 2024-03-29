import numpy as np
from scipy.special import gamma


class GGD:
    def __init__(self):
        self.alphas = np.arange(0.025, 5, 0.01)
        self.n_alphas = len(self.alphas)
        self.alphas_rev = self.alphas[::-1]
        self.gam_arr = gamma(2/self.alphas)**2 / (gamma(1/self.alphas)*gamma(3/self.alphas))

    def inverse_lookup(self, gam_hat):
        ret = np.searchsorted(self.gam_arr, gam_hat)
        if np.isscalar(ret):
            ret = min(ret, self.n_alphas-1)
        elif isinstance(ret, np.ndarray):
            ret = np.clip(ret, 0, self.n_alphas-1)
        else:
            print('Error: Expected number or array to be the result of searchssorted.')
            return None
        return ret

    def fit(self, x, return_alpha=False):
        mu = np.mean(x)
        sig = np.std(x)
        mu_abs = np.mean(np.abs(x - mu))
        if isinstance(sig, np.ndarray):
            gam_hat = mu_abs**2 / sig**2 if sig != 0 else 1
        else:
            gam_hat = np.where(sig != 0, mu_abs**2 / sig**2, np.ones_like(sig))
        alpha_ind = self.inverse_lookup(gam_hat)
        alpha = self.alphas[alpha_ind]
        b = sig / np.sqrt(gamma(3/alpha)/gamma(1/alpha))
        return b, mu, alpha if return_alpha else alpha_ind

    def pdf(x, mu, b, alpha):
        return alpha/(2*b*gamma(1/alpha)) * np.exp(-(np.abs(x - mu)/b)**alpha)


class AGGD:
    def __init__(self):
        self.alphas = np.arange(0.025, 10, 0.01)
        self.n_alphas = len(self.alphas)
        self.gam_arr = gamma(2/self.alphas)**2 / (gamma(1/self.alphas)*gamma(3/self.alphas))

    def inverse_lookup(self, gam_hat):
        ret = np.searchsorted(self.gam_arr, gam_hat)
        if np.isscalar(ret):
            ret = min(ret, self.n_alphas-1)
        elif isinstance(ret, np.ndarray):
            ret = np.clip(ret, 0, self.n_alphas-1)
        else:
            print('Error: Expected number or array to be the result of searchssorted.')
            return None
        return ret

    def pdf(x, b_l, b_r, alpha):
        return (alpha/((b_l + b_r) * gamma(1/alpha)))*np.exp(-(x/(-b_l*(x < 0) + b_r*(x >= 0)))**alpha)

    def fit(self, x, return_alpha=False):
        sig_l = np.mean(x[x < 0]**2)
        sig_r = np.mean(x[x >= 0]**2)
        mu_abs = np.mean(np.abs(x))
        mu_2 = np.mean(x**2)
        ratio = np.sqrt(sig_r/sig_l)
        gam_hat = mu_abs**2 / mu_2
        gam_hat = gam_hat * (ratio**3+1) * (ratio+1) / (ratio**2+1)**1
        alpha_ind = self.inverse_lookup(gam_hat)
        alpha = self.alphas[alpha_ind]
        b_l = np.sqrt(sig_l * gamma(3/alpha) / gamma(1/alpha))
        b_r = np.sqrt(sig_r * gamma(3/alpha) / gamma(1/alpha))
        mu = (b_l - b_r) * gamma(2/alpha) / gamma(1/alpha)
        return b_l, b_r, mu, alpha if return_alpha else alpha_ind


class MGGD:
    def __init__(self, d):
        alphas = np.arange(0.025, 10, 0.01)
        self.n_alphas = len(alphas)
        self.alphas_rev = alphas[::-1]
        self.d = d
        self.gam2_arr = d**2 * gamma(0.5*(d + 4)/self.alphas_rev) * gamma(0.5*d/self.alphas_rev) / gamma(0.5*(d + 2)/self.alphas_rev)**2
        self.c_factor_arr = d * gamma(0.5*d/self.alphas_rev) / (np.power(2, 1.0/self.alphas_rev) * gamma(0.5*(d + 2)/self.alphas_rev))
        self.entr_const_arr = (d/2) * ((1/self.alphas_rev) + np.log(np.pi) + np.log(d) - np.log(gamma(0.5*(d + 2)/self.alphas_rev))) + (1 + d/2)*np.log(gamma(0.5*d/self.alphas_rev)) - np.log(self.alphas_rev) - np.log(gamma(d/2))

    def inverse_lookup(self, kurt_hat):
        ret = np.searchsorted(self.gam2_arr, kurt_hat)
        if np.isscalar(ret):
            ret = min(ret, self.n_alphas-1)
        elif isinstance(ret, np.ndarray):
            ret = np.clip(ret, 0, self.n_alphas-1)
        else:
            print('Error: Expected number or array to be the result of searchssorted.')
            return None
        return ret

    def pdf(self, x, mu, c, alpha):
        d, n = x.shape
        if mu.ndim == 1:
            mu = np.expand_dims(mu, -1)
        diff = x - np.tile(mu, (1, n))
        c_inv = np.linalg.inv(c)
        s = np.sum((diff @ c_inv) * diff, axis=-1)
        return (gamma(d/2) * alpha)/(np.power(np.pi, d/2) * gamma(0.5*d/alpha) * np.power(2, 0.5*d/alpha) * np.sqrt(np.linalg.det(c))) * np.exp(-0.5*np.power(s, alpha))

    def fit(self, x, return_alpha=False):
        d, n = x.shape
        assert d == self.d, 'Wrong number of dimensions'
        mu = np.expand_dims(np.mean(x, 1), -1)
        sigma = np.cov(x)
        if sigma.ndim == 0:
            sigma = np.reshape(sigma, (1, 1))
        sigma_inv = np.linalg.inv(sigma)
        diff = x - np.tile(mu, (1, n))
        kurt = np.mean(np.sum((sigma_inv @ diff) * diff, axis=0)**2)
        alpha_ind = self.inverse_lookup(kurt)
        return mu, sigma, self.alphas_rev[alpha_ind] if return_alpha else alpha_ind