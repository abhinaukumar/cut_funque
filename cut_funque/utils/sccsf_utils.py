from typing import Optional, Union
import numpy as np


class CSF:
    def __init__(self) -> None:
        pass

    @staticmethod
    def have_consistent_shapes(*args) -> bool:
        is_array = [isinstance(arg, np.ndarray) for arg in args]
        shapes = [arg.shape for flag, arg in zip(is_array, args) if flag]
        return all([shape == shapes[0] for shape in shapes])

    def lms2arb(self) -> None:
        raise NotImplementedError

    def peak_frequency(self, lum: Union[float, np.ndarray], arb_dim: int) -> Union[float, np.ndarray]:
        if arb_dim < 0 or arb_dim > 2:
            raise ValueError('arb_dim must be in the range 0-2.')
        if arb_dim == 0:
            return self.arb_peak_frequency_params[0][0] * np.power(1 + self.arb_peak_frequency_params[0][1] / lum, -self.arb_peak_frequency_params[0][2])
        else:
            return self.arb_peak_frequency_params[arb_dim]

    def base_sensitivity(self, lum: Union[float, np.ndarray], arb_dim: int) -> np.ndarray:
        if arb_dim < 0 or arb_dim > 2:
            raise ValueError('arb_dim must be in the range 0-2.')
        sens = self.arb_base_sensitivity_params[arb_dim][0] * np.power(1 + self.arb_base_sensitivity_params[arb_dim][1] / lum, -self.arb_base_sensitivity_params[arb_dim][2])
        if arb_dim == 0:
            sens = sens * (1 - np.power(1 + self.arb_base_sensitivity_params[0][3] / lum, -self.arb_base_sensitivity_params[0][4]))
        return sens

    def log_parabola(self, freq: Union[float, np.ndarray], lum: Union[float, np.ndarray], arb_dim: int) -> Union[float, np.ndarray]:
        if arb_dim < 0 or arb_dim > 2:
            raise ValueError('arb_dim must be in the range 0-2.')
        if not self.have_consistent_shapes(freq, lum):
            raise ValueError('When frequency and luminance are both arrays, they must have the same shape.')

        freq_max = self.peak_frequency(lum, arb_dim)
        mask = (arb_dim > 0) and freq < freq_max
        if isinstance(mask, float):
            return 1
        else:
            return np.where(mask, 1, np.power(10, (np.log10(freq) - np.log10(freq_max))**2 / np.power(2, self.arb_log_parabola_params[arb_dim] - 1)))

    def stimulus_size_function(self, freq: Union[float, np.ndarray], area: Union[float, np.ndarray], arb_dim: int) -> Union[float, np.ndarray]:
        if arb_dim < 0 or arb_dim > 2:
            raise ValueError('arb_dim must be in the range 0-2.')
        if not self.have_consistent_shapes(freq, area):
            raise ValueError('When frequency and area are both arrays, they must have the same shape.')

        area_times_freq = np.power(area, self.arb_stimulus_exp_params[arb_dim]) * freq**2
        return np.sqrt(area_times_freq / (self.arb_stimulus_offset_params[arb_dim] + area * 0.65 + area_times_freq))
        
    def sensitivity_arb(self, freq: Union[float, np.ndarray], lum: Union[float, np.ndarray], area: Union[float, np.ndarray], arb_dim: Optional[int] = None):
        if arb_dim is not None and (arb_dim < 0 or arb_dim > 2):
            raise ValueError('arb_dim must be in the range 0-2.')
        if not self.have_consistent_shapes(freq, lum, area):
            raise ValueError('When any of frequency, luminance and area are arrays, they must all have the same shape.')

        arb_dims = [arb_dim] if arb_dim is not None else [0, 1, 2]
        sens = []
        for arb_dim in arb_dims:
            sens.append(self.base_sensitivity(lum, arb_dim) * self.stimulus_size_function(freq, area, arb_dim) / self.log_parabola(freq, lum, arb_dim))
        return sens[0] if len(sens) == 1 else np.stack(sens, axis=-1)

    def energy_function(self, freq: Union[float, np.ndarray], area: Union[float, np.ndarray], lms: np.ndarray, lms_delta: np.ndarray):
        args_to_check = [freq, area]
        if lms.shape[-1] != 3 or lms_delta.shape[-1] != 3:
            raise ValueError('Last dimension of lms and lms_delta must represent LMS channels and therefore have size 3.')
        if lms.ndim > 1:
            args_to_check.append(lms[..., 0])
        if lms_delta.ndim > 1:
            args_to_check.append(lms_delta[..., 0])
        if not self.have_consistent_shapes(*args_to_check):
            raise ValueError('Inconsistent array shapes.')

        lum = lms[..., 0] + lms[..., 1]
        sens = self.sensitivity_arb(freq, lum, area)
        arb_delta = self.lms2arb(lms, lms_delta)
        energy = np.linalg.norm(sens * arb_delta, axis=-1)
        return energy

    def detection_probability(self, freq: Union[float, np.ndarray], area: Union[float, np.ndarray], lms: np.ndarray, lms_delta: np.ndarray):
        args_to_check = [freq, area]
        if lms.shape[-1] != 3 or lms_delta.shape[-1] != 3:
            raise ValueError('Last dimension of lms and lms_delta must represent LMS channels and therefore have size 3.')
        if lms.ndim > 1:
            args_to_check.append(lms[..., 0])
        if lms_delta.ndim > 1:
            args_to_check.append(lms_delta[..., 0])
        if not self.have_consistent_shapes(*args_to_check):
            raise ValueError('Inconsistent array shapes.')
        return 1 - np.exp(np.log(0.5) * self.energy_function(freq, area, lms, lms_delta))

    def detection_threshold(self, freq: Union[float, np.ndarray], area: Union[float, np.ndarray], lms: np.ndarray, lms_delta: np.ndarray):
        return 1 / self.energy_function(freq, area, lms, lms_delta)

    def contrast_sensitivity(self, freq: Union[float, np.ndarray], area: Union[float, np.ndarray], lms: np.ndarray, lms_delta: np.ndarray):
        return self.energy_function(freq, area, lms, lms_delta) * np.sqrt(3) / np.linalg.norm(lms_delta / lms, axis=-1)


class PostReceptoralCSF(CSF):
    def __init__(self) -> None:
        self.transfer_mat_ARB = np.array([[+1.000, +3.014, +0.000],
                                          [+1.000, -2.620, +1.361],
                                          [-0.025, -0.023, +1.000]])

        self.arb_base_sensitivity_params = np.empty((3,), dtype=object)
        self.arb_base_sensitivity_params[0] = np.array([429.088, 3.99169, 0.249356, 43538.6, 0.514345])  # sA1 ... sA5
        self.arb_base_sensitivity_params[1] = np.array([1096.06, 28.7702, 0.470414])  # sR1 ... sR3
        self.arb_base_sensitivity_params[2] = np.array([37092.6, 66.2461, 0.410266])  # sB1 ... sB3

        self.arb_peak_frequency_params = np.empty((3,), dtype=object)
        self.arb_peak_frequency_params[0] = np.array([1.95966, 1427.43, 0.183791])  # rhoA1 ... rhoA3
        self.arb_peak_frequency_params[1] = 0.0370431  # rhoR
        self.arb_peak_frequency_params[2] = 0.0011757  # rhoB

        self.arb_log_parabola_params = np.array([0.931471, 3.36043, 3.92109])  # bA, bR, bB

        self.arb_stimulus_exp_params = np.array([1.1431, 1.59041, 1.47724])  # gammaA, gammaR, gammaB
        self.arb_stimulus_offset_params = np.array([65.992, 2.28102, 0.272143])  # ahatA, ahatR, ahatB

    def lms2arb(self, lms: np.ndarray, lms_delta: np.array):
        if lms.shape[-1] != 3 or lms_delta.shape[-1] != 3:
            raise ValueError('Last dimension of lms and lms_delta must represent LMS channels and therefore have size 3.')
        return (lms_delta / (lms[..., 0] + lms[..., 1])) @ self.transfer_mat_ARB.T
