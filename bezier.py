import numpy as np


def bezier(params, t):
    """Evaluate a bezier curve at time t (between 0 and 1)"""
    return np.dot(bezier_coefs(t, len(params)-1), params)


def bezier_coefs(t, order):
    """Evaluate a bezier curve at time t (between 0 and 1)"""
    if order == 0:
        return np.array([1.])
    else:
        c = bezier_coefs(t, order-1)
        return np.hstack((c, 0)) * (1.-t) + np.hstack((0, c)) * t


def bezier_deriv(params, t):
    return np.dot(bezier_deriv_coefs(t, len(params)-1), params)


def bezier_deriv_coefs(t, order):
    if order == 0:
        return np.array([0.])
    else:
        c = bezier_coefs(t, order-1)
        dc = bezier_deriv_coefs(t, order-1)
        return np.hstack((dc, 0))*(1.-t) + np.hstack((0, dc))*t - np.hstack((c, 0)) + np.hstack((0, c))


def bezier_second_deriv(params, t):
    return np.dot(bezier_second_deriv_coefs(t, len(params)-1), params)


def bezier_second_deriv_coefs(t, order):
    if order == 0:
        return np.array([0.])
    else:
        dc = bezier_deriv_coefs(t, order-1)
        ddc = bezier_second_deriv_coefs(t, order-1)
        return np.hstack((ddc, 0))*(1.-t) + np.hstack((0, ddc))*t - np.hstack((dc, 0))*2 + np.hstack((0, dc))*2


def repeat_diag(x, k):
    return np.hstack([np.eye(k, dtype=x.dtype) * xi for xi in x])


def zero_offset_bezier(params, t):
    return np.dot(zero_offset_bezier_coefs(t, len(params)), params)


def zero_offset_bezier_coefs(t, order):
    return bezier_coefs(t, order)[1:]


def zero_offset_bezier_mat(t, order, ndims):
    return repeat_diag(zero_offset_bezier_coefs(t, order), ndims)


def zero_offset_bezier_deriv(params, t):
    return np.dot(zero_offset_bezier_deriv_coefs(t, len(params)), params)


def zero_offset_bezier_deriv_coefs(t, order):
    return bezier_deriv_coefs(t, order)[1:]


def zero_offset_bezier_second_deriv(params, t):
    return np.dot(zero_offset_bezier_second_deriv_coefs(t, len(params)), params)


def zero_offset_bezier_second_deriv_coefs(t, order):
    return bezier_second_deriv_coefs(t, order)[1:]


def zero_offset_bezier_second_deriv_mat(t, order, ndims):
    return repeat_diag(zero_offset_bezier_second_deriv_coefs(t, order), ndims)
