from typing import Optional
import numpy as np


def get_transfer_mat_lms() -> np.ndarray:
    transfer_mat_lms = np.array([[+0.187596268556126, +0.585168649077728, -0.026384263306304],
                                 [-0.133397430663221, +0.405505777260049, +0.034502127690364],
                                 [+0.000244379021663, -0.000542995890619, +0.019406849066323]])
    return transfer_mat_lms


def get_transfer_mat_dkl(xyz_white: Optional[np.ndarray] = np.array([0.31271, 0.32902, 0.35827])) -> np.ndarray:
    if xyz_white.shape != (3,):
        raise ValueError('Expected xyz_white to be an array of shape (3,)')
    transfer_mat_lms = get_transfer_mat_lms()
    xyz_white = xyz_white / np.sum(xyz_white)
    lms_white = xyz_white @ transfer_mat_lms.T
    mc1 = lms_white[0] / lms_white[1]
    mc2 = (lms_white[0] + lms_white[1]) / lms_white[2]
    transfer_mat_dkl = np.array([[+1.0, +1.0, +0.0],
                                 [+1.0, -mc1, +0.0],
                                 [-1.0, -1.0, +mc2]])
    return transfer_mat_dkl
