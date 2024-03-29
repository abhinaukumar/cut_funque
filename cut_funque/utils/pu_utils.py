import numpy as np

from .sccsf_utils import PostReceptoralCSF
from videolib.nonlinearities import five_param_nonlinearity
from .transfer_utils import get_transfer_mat_dkl


def seven_param_nonlinearity(lums, params):
    assert(len(params)) == 7
    return params[6]*(five_param_nonlinearity(lums, params[:5]) - params[5])


lms_white = np.array([0.7347453, 0.31628247, 0.02082004])
pq_params = np.array([107/128, 2413/128, 2392/128, 1305/8192, 2523/32, 0, 0])


pucolor_params = np.array([[-2.79942827e+01, 5.43021183e+02, 2.93417157e-01, 3.19756117e-01, 1.05592639e+00],
                              [-1.23250134e+01, 1.20468545e+03, 3.76393010e-02, 4.97095626e-01, 8.43942000e-01],
                              [-1.05126925e+00, 7.12940329e+01, 4.46580895e-02, 4.54908601e-01, 9.52573208e-01]])
pucolor_offset = np.array([8.82066996e+01, 1.03596365e-03, -3.36853641e-04])
pucolor_scale = 0.0004386986748681211

csf = PostReceptoralCSF()
transfer_mat_dkl = get_transfer_mat_dkl()
inv_transfer_mat_dkl = np.linalg.inv(transfer_mat_dkl)


def pucolor(x):
    ints = []
    lum = x[..., 0] + x[..., 1]
    lum = np.clip(lum, 5e-3, 1e4)
    for lms_ind in range(3):
        ints.append(five_param_nonlinearity(lum, pucolor_params[lms_ind]) / lum)
    ints = np.stack(ints, axis=-1)
    y = ints * (x @ transfer_mat_dkl.T)
    return (y - pucolor_offset) * pucolor_scale
