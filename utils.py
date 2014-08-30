import numpy as np
from lie import skew


def cayley_mat(s):
    s = np.asarray(s, float)
    return np.eye(3) * (1. - np.dot(s, s)) + 2.*skew(s) + 2.*np.outer(s, s)


def cayley_denom(s):
    s = np.asarray(s, float)
    return 1. + np.dot(s, s)


def cayley(s):
    s = np.asarray(s, float)
    return cayley_mat(s) / cayley_denom(s)


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
