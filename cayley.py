import numpy as np
from lie import SO3, skew


def cayley_mat(s):
    s = np.asarray(s, float)
    return np.eye(3) * (1. - np.dot(s, s)) + 2.*skew(s) + 2.*np.outer(s, s)


def cayley_denom(s):
    s = np.asarray(s, float)
    return 1. + np.dot(s, s)


def cayley(s):
    s = np.asarray(s, float)
    return cayley_mat(s) / cayley_denom(s)


def cayley_inv(r):
    w = SO3.log(r)
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return w
    else:
        return w * np.tan(theta / 2.) / theta


def cayley_av_mat(x):
    return (np.eye(3) - skew(x)) * 2. / (1. + np.dot(x, x))


def angular_velocity_from_cayley_deriv(x, dx):
    return np.dot(cayley_av_mat(x), dx)
