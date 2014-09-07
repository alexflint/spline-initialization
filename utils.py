import numpy as np
from lie import skew, SO3


def skew_jacobian():
    """Return the jacobian of flatten(skew(x)) with respect to x."""
    return np.array([[0,  0,  0],
                     [0,  0, -1],
                     [0,  1,  0],
                     [0,  0,  1],
                     [0,  0,  0],
                     [-1, 0,  0],
                     [0, -1,  0],
                     [1,  0,  0],
                     [0,  0,  0]])


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


def normalized(x):
    x = np.asarray(x, float)
    return x / np.linalg.norm(x)


def essential_matrix(R1, p1, R2, p2):
    Rrel = np.dot(R2, R1.T)
    prel = np.dot(R1, p2-p1)
    return essential_matrix_from_relative_pose(Rrel, prel)


def essential_matrix_from_relative_pose(Rrel, prel):
    return np.dot(Rrel, skew(prel))


def add_white_noise(x, sigma):
    return x + np.random.randn(*x.shape) * sigma


def add_orientation_noise(x, sigma):
    x = np.atleast_3d(x)
    return np.array([np.dot(xi, SO3.exp(np.random.randn(3)*sigma)) for xi in x])
