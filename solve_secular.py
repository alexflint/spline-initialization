import numpy as np
import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt

from lie import SO3
from utils import normalized


def main():
    a = SO3.exp(np.random.rand(3))
    b = np.array((2, 2, 2))

    norm_x = 1
    true_x = np.dot(a.T, normalized(b))

    u, s, vt = np.linalg.svd(a)
    v = vt.T
    btilde = np.dot(u.T, b)

    def secular(k):
        return np.sum(np.square(s*btilde / (s*s + k))) - norm_x*norm_x

    k = scipy.optimize.fsolve(secular, 1.)

    estimated_x = np.dot(v, s*btilde / (s*s + k))
    print estimated_x
    print np.dot(a, estimated_x)


if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(suppress=True, linewidth=500)
    main()
