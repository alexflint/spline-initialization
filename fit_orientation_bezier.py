import os
import numpy as np
import scipy.optimize
import numdifftools

from bezier import bezier, zero_offset_bezier, zero_offset_bezier_deriv
from utils import skew, cayley, angular_velocity_from_cayley_deriv
from lie import SO3

import matplotlib.pyplot as plt


def predict_gyro(bezier_params, time, gyro_bias):
    orient = cayley(zero_offset_bezier(bezier_params, time))
    orient_deriv = numdifftools.Jacobian(lambda t: SO3.log(cayley(zero_offset_bezier(bezier_params, t))))
    return np.dot(orient, np.squeeze(orient_deriv(time))) + gyro_bias


def gyro_residual(bezier_params, gyro_timestamp, gyro_bias, gyro_reading):
    return predict_gyro(bezier_params, gyro_timestamp, gyro_bias) - gyro_reading


def gyro_residuals(bezier_params, gyro_timestamps, gyro_bias, gyro_readings):
    assert len(gyro_timestamps) == len(gyro_readings)
    return np.hstack([gyro_residual(bezier_params, t, gyro_bias, r)
                     for t, r in zip(gyro_timestamps, gyro_readings)])

def angular_velocity_left(f, t, step=1e-8):
    return SO3.log(np.dot(f(t + step), f(t).T)) / step


def angular_velocity_right(f, t, step=1e-8):
    return SO3.log(np.dot(f(t).T, f(t + step))) / step


def run_furgale():
    bezier_order = 4

    bezier_params = np.random.rand(bezier_order, 3)
    bezier_params *= 10
    bez = lambda t: bezier(bezier_params, t)
    bezderiv = lambda t: np.squeeze(numdifftools.Jacobian(lambda tt: bez(tt))(t))

    t0 = 1.23
    r0 = cayley(bez(t0))
    w0 = angular_velocity_from_cayley_deriv(bez(t0), bezderiv(t0))

    print 'Params:'
    print bezier_params
    print 'Rotation'
    print r0

    print 'Numeric right:', angular_velocity_right(lambda t: cayley(bez(t)), t0)
    print 'Analytic global:', w0

    print 'Numeric left:', angular_velocity_left(lambda t: cayley(bez(t)), t0)
    print 'Analytic local:', np.dot(r0, w0)


def run_derivative_test():
    bezier_order = 4

    bezier_params = np.random.rand(bezier_order, 3)
    bezier_params[0] *= 10

    print bezier_params

    print SO3.exp(bezier(bezier_params, 0.))

    t0 = 1.42
    w0 = zero_offset_bezier(bezier_params, t0)
    wderiv = np.squeeze(numdifftools.Jacobian(lambda t: bezier(bezier_params, t))(t0))
    R0 = SO3.exp(w0)

    print 'Numeric:', angular_velocity_local(lambda t: SO3.exp(bezier(bezier_params, t)), t0)
    print 'w0:', w0
    print 'wderiv:', wderiv
    print 'Analytic:'
    print np.cross(wderiv, w0)
    print np.dot(R0, np.cross(wderiv, w0))
    print np.dot(R0.T, np.cross(wderiv, w0))


def run_optimize_via_finite_diffs():
    bezier_order = 2
    num_gyro_readings = 6

    #path = os.path.expanduser('~/Data/Initialization/closed_flat/gyro.txt')
    #gyro_data = np.loadtxt(path)
    #gyro_timestamps = gyro_data[:,0]
    #gyro_readings = gyro_data[:,1:]

    true_gyro_timestamps = np.linspace(0, 1, num_gyro_readings)
    true_params = np.random.rand(bezier_order, 3)
    true_gyro_bias = np.random.rand(3)
    true_gyro_readings = np.array([predict_gyro(true_params, t, true_gyro_bias) for t in true_gyro_timestamps])

    observed_gyro_timestamps = true_gyro_timestamps
    observed_gyro_readings = true_gyro_readings

    seed_params = np.zeros((bezier_order, 3))

    def res(params):
        params = params.reshape((bezier_order, 3))
        return gyro_residuals(params, observed_gyro_timestamps, true_gyro_bias, observed_gyro_readings)

    def cost(params):
        residuals = res(params)
        return np.dot(residuals, residuals)

    def on_step(cur):
        print 'Step'

    out = scipy.optimize.minimize(cost,
                                  seed_params.flatten(),
                                  tol=1,
                                  options=dict(maxiter=500),
                                  callback=on_step)

    estimated_params = out.x.reshape((bezier_order, 3))

    print '\nTrue params:'
    print true_params

    print '\nEstimated params:'
    print estimated_params

    print '\nCost at seed:', cost(seed_params.flatten())
    print 'Cost at estimate:', cost(out.x)

    estimated_gyro_readings = np.array([predict_gyro(estimated_params, t, true_gyro_bias)
                                        for t in observed_gyro_timestamps])

    plt.plot(observed_gyro_timestamps, observed_gyro_readings, '-')
    plt.plot(observed_gyro_timestamps, estimated_gyro_readings, ':')
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(suppress=True)
    #run_optimize_via_finite_diffs()
    #run_derivative_test()
    run_furgale()
