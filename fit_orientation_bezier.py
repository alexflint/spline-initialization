import numpy as np
import scipy.optimize
import numdifftools

from bezier import bezier, zero_offset_bezier, zero_offset_bezier_deriv
from utils import skew, cayley, angular_velocity_from_cayley_deriv
from lie import SO3

import matplotlib
import matplotlib.pyplot as plt


def predict_orientation(bezier_params, time):
    return cayley(zero_offset_bezier(bezier_params, time))


def predict_gyro(bezier_params, gyro_bias, time):
    s = zero_offset_bezier(bezier_params, time)
    s_deriv = zero_offset_bezier_deriv(bezier_params, time)
    orient = cayley(s)
    angular_velocity = angular_velocity_from_cayley_deriv(s, s_deriv)
    return np.dot(orient, angular_velocity) + gyro_bias


def gyro_residual(bezier_params, gyro_bias, gyro_timestamp, gyro_reading):
    return predict_gyro(bezier_params, gyro_bias, gyro_timestamp) - gyro_reading


def gyro_residuals(bezier_params, gyro_bias, gyro_timestamps, gyro_readings):
    assert len(gyro_timestamps) == len(gyro_readings)
    return np.hstack([gyro_residual(bezier_params, gyro_bias, t, r)
                     for t, r in zip(gyro_timestamps, gyro_readings)])

def angular_velocity_left(f, t, step=1e-8):
    return SO3.log(np.dot(f(t + step), f(t).T)) / step


def angular_velocity_right(f, t, step=1e-8):
    return SO3.log(np.dot(f(t).T, f(t + step))) / step


def orientation_residuals(bezier_params, observed_timestamps, observed_orientations):
    return np.hstack([SO3.log(np.dot(predict_orientation(bezier_params, t).T, r))
                      for t, r in zip(observed_timestamps, observed_orientations)])


def add_white_noise(x, sigma):
    return x + np.random.randn(*x.shape) * sigma


def add_orientation_noise(x, sigma):
    x = np.atleast_3d(x)
    return np.array([np.dot(xi, SO3.exp(np.random.randn(3)*sigma)) for xi in x])


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


def run_optimize():
    bezier_order = 3
    num_gyro_readings = 50
    num_frames = 5

    frame_timestamp_noise = 1e-3
    frame_orientation_noise = .02
    gyro_timestamp_noise = 1e-3
    gyro_noise = .01

    #path = os.path.expanduser('~/Data/Initialization/closed_flat/gyro.txt')
    #gyro_data = np.loadtxt(path)
    #gyro_timestamps = gyro_data[:,0]
    #gyro_readings = gyro_data[:,1:]

    true_gyro_timestamps = np.linspace(0, 1, num_gyro_readings)
    true_params = np.random.rand(bezier_order, 3)
    true_gyro_bias = np.random.rand(3)
    true_gyro_readings = np.array([predict_gyro(true_params, true_gyro_bias, t)
                                   for t in true_gyro_timestamps])

    true_frame_timestamps = np.linspace(0, 1, num_frames)
    true_frame_orientations = np.array([predict_orientation(true_params, t) for t in true_frame_timestamps])

    observed_gyro_timestamps = add_white_noise(true_gyro_timestamps, gyro_timestamp_noise)
    observed_gyro_readings = add_white_noise(true_gyro_readings, gyro_noise)
    observed_frame_timestamps = add_white_noise(true_frame_timestamps, frame_timestamp_noise)
    observed_frame_orientations = add_orientation_noise(true_frame_orientations, frame_orientation_noise)

    seed_params = np.zeros((bezier_order, 3))
    seed_gyro_bias = np.zeros(3)
    seed = np.hstack((seed_gyro_bias, seed_params.flatten()))

    def residuals(x):
        gyro_bias = x[:3]
        bezier_params = x[3:].reshape((bezier_order, 3))
        r_gyro = gyro_residuals(bezier_params, gyro_bias, observed_gyro_timestamps, observed_gyro_readings)
        r_orient = orientation_residuals(bezier_params, observed_frame_timestamps, observed_frame_orientations)
        return np.hstack((r_gyro, r_orient))

    def cost(x):
        r = residuals(x)
        return np.dot(r, r)

    print 'Optimizing...'
    out = scipy.optimize.minimize(cost,
                                  seed,
                                  tol=1e-8,
                                  options=dict(maxiter=500))

    estimate = out.x
    estimated_gyro_bias = estimate[:3]
    estimated_params = estimate[3:].reshape((bezier_order, 3))

    print '\nTrue params:'
    print true_params

    print '\nEstimated params:'
    print estimated_params

    print '\nTrue gyro bias:'
    print true_gyro_bias

    print '\nEstimated gyro bias:'
    print estimated_gyro_bias

    print '\nCost at seed:', cost(seed)
    print 'Cost at estimate:', cost(estimate)

    plot_timestamps = np.linspace(0, 1, 50)

    estimated_gyro_readings = np.array([predict_gyro(estimated_params, true_gyro_bias, t)
                                        for t in plot_timestamps])

    true_orientations = np.array([SO3.log(predict_orientation(true_params, t))
                                  for t in plot_timestamps])
    observed_orientations = np.array(map(SO3.log, observed_frame_orientations))
    estimated_orientations = np.array([SO3.log(predict_orientation(estimated_params, t))
                                       for t in plot_timestamps])

    plt.figure(1)
    plt.plot(true_gyro_timestamps, true_gyro_readings, '-', label='true')
    plt.plot(true_gyro_timestamps, observed_gyro_readings, 'x', label='observed')
    plt.plot(plot_timestamps, estimated_gyro_readings, ':', label='estimated')
    plt.xlim(-.1, 1.5)
    plt.legend()

    plt.figure(2)
    plt.plot(plot_timestamps, true_orientations, '-', label='true')
    plt.plot(true_frame_timestamps, observed_orientations, 'x', label='observed')
    plt.plot(plot_timestamps, estimated_orientations, ':', label='estimated')
    plt.xlim(-.1, 1.5)
    plt.legend()

    plt.show()

if __name__ == '__main__':
    np.random.seed(1)
    np.set_printoptions(suppress=True)
    matplotlib.rc('font', size=9)
    matplotlib.rc('legend', fontsize=9)

    run_optimize()
    #run_derivative_test()
    #run_furgale()
