import torch
import numpy as np
import math

def sinusoid_func(input_arr: torch.Tensor, freq_param=1.0, scale_param=1, skip_torch=False, cos=False):
    """
    :param freq_param: multiplier for sine frequency
    """
    func = np.sin if not cos else np.cos
    if skip_torch:
        return func(input_arr * freq_param) * scale_param
    else:
        return func(torch.Tensor(input_arr) * freq_param) * scale_param


def sum_of_sinusoids(input_arr: torch.Tensor, freq_params, scale_params, skip_torch=False, cos=False):
    """
    :param sine_multiplier: multiplier for sine frequency
    """
    if skip_torch:
        return np.sum(np.stack([sinusoid_func(input_arr, 2*math.pi*freq, scale, skip_torch=True, cos=cos) for freq, scale in zip(freq_params, scale_params)], dim=0), dim=0)
    else:
        return torch.stack([sinusoid_func(input_arr, 2*math.pi*freq, scale, cos=cos) for freq, scale in zip(freq_params, scale_params)], dim=0).sum(dim=0)


def clipped_exp_sns(input_arr, shift_param=0.0, scale_param=1.0, min_clip_param=None, max_clip_param=None, skip_torch=False):
    """
    :param shift_param: at what input value does exp value go to 1
    :param scale_param: scale the exponential
    :param clip_param: supply int if desired to clip exp to a certain max value.
    """
    if skip_torch:
        op_arr = scale_param*np.exp(input_arr - shift_param)
    else:
        op_arr = scale_param*torch.exp(torch.Tensor(input_arr) - shift_param)
    if min_clip_param is not None:
        if skip_torch:
            op_arr = np.minimum(np.maximum(op_arr, min_clip_param), max_clip_param)
        else:
            op_arr = torch.minimum(torch.maximum(op_arr, torch.tensor(min_clip_param)), torch.tensor(max_clip_param))
    return op_arr


def polynomial_1d(coefficients, inputs):
    """
    Evaluate a polynomial with given coefficients on an array of inputs.

    Args:
        coefficients (array-like): Coefficients of the polynomial, starting with the
            coefficient for the highest degree term and ending with the coefficient for
            the constant term.
        inputs (array-like): Inputs at which to evaluate the polynomial.

    Returns:
        outputs (ndarray): Outputs of the polynomial at the given inputs.
    """
    # Create a polynomial object from the coefficients
    polynomial = np.poly1d(coefficients)

    # Evaluate the polynomial at the given inputs
    outputs = polynomial(inputs)

    return outputs
