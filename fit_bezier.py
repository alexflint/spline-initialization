import numpy as np
from bezier import bezier, bezier_coefs, zero_offset_bezier, zero_offset_bezier_coefs

import matplotlib
import matplotlib.pyplot as plt


def fit_bezier_1d(ts, ys, bezier_order):
    jacobian = np.array([bezier_coefs(t, bezier_order) for t in ts])
    residual = np.array([bezier(np.zeros(bezier_order), t) - y for t, y in zip(ts, ys)])
    return np.linalg.lstsq(jacobian, -residual)[0]


def fit_bezier(ts, ys, bezier_order):
    ys = np.asarray(ys)
    if ys.ndim == 1:
        return fit_bezier_1d(ts, ys, bezier_order)
    else:
        return np.hstack([fit_bezier_1d(ts, ys[:, i], bezier_order)[:, np.newaxis]
                          for i in range(ys.shape[1])])


def fit_zero_offset_bezier_1d(ts, ys, bezier_order):
    jacobian = np.array([zero_offset_bezier_coefs(t, bezier_order) for t in ts])
    residual = np.array([zero_offset_bezier(np.zeros(bezier_order), t) - y for t, y in zip(ts, ys)])
    return np.linalg.lstsq(jacobian, -residual)[0]


def fit_zero_offset_bezier(ts, ys, bezier_order):
    ys = np.asarray(ys)
    if ys.ndim == 1:
        return fit_zero_offset_bezier_1d(ts, ys, bezier_order)
    else:
        return np.hstack([fit_zero_offset_bezier_1d(ts, ys[:, i], bezier_order)[:, np.newaxis]
                          for i in range(ys.shape[1])])


def main():
    bezier_order = 4
    num_samples = 10

    true_controls = np.random.rand(bezier_order, 3)
    true_ts = np.linspace(0, 1, num_samples)
    true_ys = np.array([zero_offset_bezier(true_controls, t) for t in true_ts])

    estimated_controls = fit_zero_offset_bezier(true_ts, true_ys, bezier_order)

    plot_ts = np.linspace(0, 1, 50)
    plot_true_ys = np.array([zero_offset_bezier(true_controls, t) for t in plot_ts])
    estimated_ys = np.array([zero_offset_bezier(estimated_controls, t) for t in plot_ts])

    plt.clf()
    plt.plot()
    plt.plot(plot_ts, estimated_ys, ':', alpha=1, label='estimated')
    plt.plot(plot_ts, plot_true_ys, '-', alpha=.3, label='true')
    plt.plot(true_ts, true_ys, 'x', alpha=1, label='observed')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    np.set_printoptions(linewidth=500, suppress=True)
    matplotlib.rc('font', size=9)
    matplotlib.rc('legend', fontsize=9)

    main()
