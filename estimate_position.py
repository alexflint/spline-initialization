import numdifftools

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from lie import SO3
from geometry import pr
from cayley import cayley
from utils import normalized, skew_jacobian, essential_matrix, add_white_noise, add_orientation_noise
from bezier import zero_offset_bezier, zero_offset_bezier_mat, zero_offset_bezier_second_deriv, zero_offset_bezier_second_deriv_mat


def diagify(x, k):
    x = np.atleast_2d(x)
    m, n = x.shape
    out = np.zeros((m*k, n*k), x.dtype)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i*k:i*k+k, j*k:j*k+k] = np.eye(k) * x[i, j]
    return out


def dots(*args):
    return reduce(np.dot, args)


def accel_residual(pos_controls, accel_bias, gravity,
                   timestamp, accel_reading, orientation):
    global_accel = zero_offset_bezier_second_deriv(pos_controls, timestamp)
    apparent_accel = np.dot(orientation, global_accel + gravity) + accel_bias
    return apparent_accel - accel_reading


def accel_jacobian(bezier_order, timestamp, orientation):
    bezier_mat = zero_offset_bezier_second_deriv_mat(timestamp, bezier_order, 3)
    return np.hstack((np.dot(orientation, bezier_mat), np.eye(3), orientation))


def evaluate_accel_residuals(pos_controls, accel_bias, gravity,
                             accel_timestamps, accel_readings, accel_orientations):
    return np.hstack([accel_residual(pos_controls, accel_bias, gravity, t, accel, R)
                     for t, R, accel in zip(accel_timestamps, accel_orientations, accel_readings)])


def evaluate_accel_jacobians(bezier_order, accel_timestamps, accel_orientations):
    return np.vstack([accel_jacobian(bezier_order, t, R)
                      for t, R in zip(accel_timestamps, accel_orientations)])


def epipolar_residual(pos_controls, ti, tj, zi, zj, Ri, Rj):
    pi = zero_offset_bezier(pos_controls, ti)
    pj = zero_offset_bezier(pos_controls, tj)
    E = essential_matrix(Ri, pi, Rj, pj)
    return dots(zj, E, zi)


def epipolar_jacobian(bezier_order, ti, tj, zi, zj, Ri, Rj):
    Rrel = np.dot(Rj, Ri.T)
    zzt = np.outer(zj, zi).flatten()
    Ai = zero_offset_bezier_mat(ti, bezier_order, 3)
    Aj = zero_offset_bezier_mat(tj, bezier_order, 3)
    return dots(zzt, diagify(Rrel, 3), skew_jacobian(), np.dot(Ri, Aj - Ai))


def evaluate_epipolar_residuals(pos_controls, frame_timestamps, frame_orientations,
                                features, feature_mask=None):
    residuals = []
    for i, (ti, Ri) in enumerate(zip(frame_timestamps, frame_orientations)):
        for j, (tj, Rj) in enumerate(zip(frame_timestamps, frame_orientations)):
            if i != j:
                for k in range(features.shape[1]):
                    if feature_mask is None or (feature_mask[i, k] and feature_mask[j, k]):
                        zi = features[i][k]
                        zj = features[j][k]
                        residuals.append(epipolar_residual(pos_controls, ti, tj, zi, zj, Ri, Rj))
    return np.array(residuals)


def evaluate_epipolar_jacobians(bezier_order, frame_timestamps, frame_orientations,
                                features, feature_mask=None):
    jacobians = []
    for i, (ti, Ri) in enumerate(zip(frame_timestamps, frame_orientations)):
        for j, (tj, Rj) in enumerate(zip(frame_timestamps, frame_orientations)):
            if i != j:
                for k in range(features.shape[1]):
                    if feature_mask is None or (feature_mask[i, k] and feature_mask[j, k]):
                        zi = features[i][k]
                        zj = features[j][k]
                        jacobians.append(epipolar_jacobian(bezier_order, ti, tj, zi, zj, Ri, Rj))
    return np.array(jacobians)


def run_accel_finite_differences():
    np.random.seed(0)

    bezier_order = 4
    pos_controls = np.random.randn(bezier_order, 3)
    accel_bias = np.random.randn(3)
    gravity = np.random.randn(3)
    a = np.random.randn(3)
    R = SO3.exp(np.random.randn(3))
    t = .5

    def r(delta):
        k = bezier_order * 3
        assert len(delta) == k + 6
        return accel_residual(pos_controls + delta[:k].reshape((bezier_order, 3)),
                              accel_bias + delta[k:k],
                              gravity + delta[k+3:k+6],
                              t,
                              a,
                              R)

    J_numeric = numdifftools.Jacobian(r)(np.zeros(bezier_order*3+6))
    J_analytic = accel_jacobian(bezier_order, t, R)

    print '\nNumeric:'
    print J_numeric

    print '\nAnalytic:'
    print J_analytic

    np.testing.assert_array_almost_equal(J_numeric, J_analytic, decimal=8)


def run_epipolar_finite_differences():
    np.random.seed(0)

    bezier_order = 4
    pos_controls = np.random.randn(bezier_order, 3)
    ti, tj = np.random.randn(2)
    zi, zj = np.random.randn(2, 3)
    Ri = SO3.exp(np.random.randn(3))
    Rj = SO3.exp(np.random.randn(3))

    def r(delta):
        assert len(delta) == bezier_order * 3
        return epipolar_residual(pos_controls + delta.reshape((bezier_order, 3)),
                                 ti, tj, zi, zj, Ri, Rj)

    J_numeric = np.squeeze(numdifftools.Jacobian(r)(np.zeros(bezier_order*3)))
    J_analytic = epipolar_jacobian(bezier_order, ti, tj, zi, zj, Ri, Rj)

    print '\nNumeric:'
    print J_numeric

    print '\nAnalytic:'
    print J_analytic

    np.testing.assert_array_almost_equal(J_numeric, J_analytic, decimal=8)


def estimate_position(bezier_degree,
                      observed_accel_timestamps,
                      observed_accel_orientations,
                      observed_accel_readings,
                      observed_frame_timestamps,
                      observed_frame_orientations,
                      observed_features):
    # Compute offsets for parameter vector
    position_offs = 0
    accel_bias_offset = position_offs + (bezier_degree-1)*3
    gravity_offset = accel_bias_offset + 3

    # Setup linear system
    accel_res = evaluate_accel_residuals(np.zeros((bezier_degree-1, 3)), np.zeros(3), np.zeros(3),
                                         observed_accel_timestamps, observed_accel_readings, observed_accel_orientations)
    accel_jac = evaluate_accel_jacobians(bezier_degree-1, observed_accel_timestamps, observed_accel_orientations)

    epipolar_res = evaluate_epipolar_residuals(np.zeros((bezier_degree-1, 3)), observed_frame_timestamps,
                                               observed_frame_orientations, observed_features)
    epipolar_jac = evaluate_epipolar_jacobians(bezier_degree-1, observed_frame_timestamps,
                                               observed_frame_orientations, observed_features)
    epipolar_jac = np.hstack((epipolar_jac, np.zeros((epipolar_jac.shape[0], 6))))

    residual = np.hstack((accel_res, epipolar_res))
    jacobian = np.vstack((accel_jac, epipolar_jac))

    # Solve
    jtj = np.dot(jacobian.T, jacobian)
    jtr = np.dot(jacobian.T, residual)
    estimated_vars = np.squeeze(np.linalg.solve(jtj, -jtr))

    # Unpack result and compute error
    estimated_pos_controls = np.reshape(estimated_vars[position_offs:position_offs+(bezier_degree-1)*3], (bezier_degree-1, 3))
    estimated_accel_bias = np.asarray(estimated_vars[accel_bias_offset:accel_bias_offset+3])
    estimated_gravity = np.asarray(estimated_vars[gravity_offset:gravity_offset+3])
    return estimated_pos_controls, estimated_accel_bias, estimated_gravity


def predict_accel(pos_controls, orient_controls, accel_bias, gravity, t):
    orientation = cayley(zero_offset_bezier(orient_controls, t))
    global_accel = zero_offset_bezier_second_deriv(pos_controls, t)
    return np.dot(orientation, global_accel + gravity) + accel_bias


def predict_feature(pos_controls, orient_controls, landmark, t):
    r = cayley(zero_offset_bezier(orient_controls, t))
    p = zero_offset_bezier(pos_controls, t)
    return normalized(np.dot(r, landmark - p))


def run_position_estimation():
    #
    # Construct ground truth
    #
    num_frames = 5
    num_landmarks = 150
    num_imu_readings = 80
    bezier_degree = 4

    print 'Num landmarks:', num_landmarks
    print 'Num frames:', num_frames
    print 'Num IMU readings:', num_imu_readings
    print 'Bezier curve degree:', bezier_degree

    # Both splines should start at 0,0,0
    true_frame_timestamps = np.linspace(0, .9, num_frames)
    true_accel_timestamps = np.linspace(0, 1, num_imu_readings)

    true_rot_controls = np.random.randn(bezier_degree-1, 3)
    true_pos_controls = np.random.randn(bezier_degree-1, 3)

    true_landmarks = np.random.randn(num_landmarks, 3) * 20

    true_frame_orientations = np.array([cayley(zero_offset_bezier(true_rot_controls, t)) for t in true_frame_timestamps])
    true_frame_positions = np.array([zero_offset_bezier(true_pos_controls, t) for t in true_frame_timestamps])

    true_gravity_magnitude = 9.8
    true_gravity = normalized(np.random.rand(3)) * true_gravity_magnitude
    true_accel_bias = np.random.randn(3)

    true_imu_orientations = np.array([cayley(zero_offset_bezier(true_rot_controls, t)) for t in true_accel_timestamps])
    true_accel_readings = np.array([predict_accel(true_pos_controls, true_rot_controls, true_accel_bias, true_gravity, t)
                                    for t in true_accel_timestamps])

    true_features = np.array([[predict_feature(true_pos_controls, true_rot_controls, x, t) for x in true_landmarks]
                              for t in true_frame_timestamps])

    print np.min(true_features.reshape((-1, 3)), axis=0)
    print np.max(true_features.reshape((-1, 3)), axis=0)

    #
    # Add sensor noise
    #

    accel_timestamp_noise = 0
    accel_reading_noise = 0#0.1
    accel_orientation_noise = 0#0.001

    frame_timestamp_noise = 0
    frame_orientation_noise = 0#0.001
    feature_noise = 0.001

    observed_accel_timestamps = add_white_noise(true_accel_timestamps, accel_timestamp_noise)
    observed_accel_readings = add_white_noise(true_accel_readings, accel_reading_noise)
    observed_accel_orientations = add_orientation_noise(true_imu_orientations, accel_orientation_noise)

    observed_frame_timestamps = add_white_noise(true_frame_timestamps, frame_timestamp_noise)
    observed_frame_orientations = add_orientation_noise(true_frame_orientations, frame_orientation_noise)
    observed_features = add_white_noise(true_features, feature_noise)

    #
    # Solve
    #

    estimated_pos_controls, estimated_accel_bias, estimated_gravity = estimate_position(
        bezier_degree,
        observed_accel_timestamps,
        observed_accel_orientations,
        observed_accel_readings,
        observed_frame_timestamps,
        observed_frame_orientations,
        observed_features)

    estimated_positions = np.array([zero_offset_bezier(estimated_pos_controls, t) for t in true_frame_timestamps])
    re_estimated_gravity = normalized(estimated_gravity) * true_gravity_magnitude

    estimated_accel_readings = np.array([predict_accel(estimated_pos_controls, true_rot_controls, estimated_accel_bias, estimated_gravity, t)
                                         for t in true_accel_timestamps])

    estimated_pfeatures = np.array([[pr(predict_feature(estimated_pos_controls, true_rot_controls, x, t)) for x in true_landmarks]
                                    for t in true_frame_timestamps])
    true_pfeatures = pr(true_features)
    observed_pfeatures = pr(observed_features)

    #
    # Report
    #

    print 'Accel bias error:', np.linalg.norm(estimated_accel_bias - true_accel_bias)
    print '  True accel bias:', true_accel_bias
    print '  Estimated accel bias:', estimated_accel_bias

    print 'Gravity error:', np.linalg.norm(estimated_gravity - true_gravity)
    print '  True gravity:', true_gravity
    print '  Estimated gravity:', estimated_gravity
    print '  Estimated gravity magnitude:', np.linalg.norm(estimated_gravity)
    print '  Re-normalized gravity error: ', np.linalg.norm(re_estimated_gravity - true_gravity)
    for i in range(num_frames):
        print 'Frame %d position error: %f' % (i, np.linalg.norm(estimated_positions[i] - true_frame_positions[i]))

    fig = plt.figure(1, figsize=(14, 10))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ts = np.linspace(0, 1, 100)
    true_ps = np.array([zero_offset_bezier(true_pos_controls, t) for t in ts])
    estimated_ps = np.array([zero_offset_bezier(estimated_pos_controls, t) for t in ts])
    ax.plot(true_ps[:, 0], true_ps[:, 1], true_ps[:, 2], '-b')
    ax.plot(estimated_ps[:, 0], estimated_ps[:, 1], estimated_ps[:, 2], '-r')

    #ax.plot(true_landmarks[:,0], true_landmarks[:,1], true_landmarks[:,2], '.k')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(true_accel_timestamps, true_accel_readings, '-', label='true')
    ax.plot(observed_accel_timestamps, observed_accel_readings, 'x', label='observed')
    ax.plot(true_accel_timestamps, estimated_accel_readings, ':', label='estimated')
    ax.legend()
    ax.set_xlim(-.1, 1.5)

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(true_pfeatures[1, :, 0], true_pfeatures[1, :, 1], 'x', label='true', alpha=.8)
    ax.plot(estimated_pfeatures[1, :, 0], estimated_pfeatures[1, :, 1], 'o', label='estimated', alpha=.4)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(true_pfeatures[-1, :, 0], true_pfeatures[-1, :, 1], '.', label='true', alpha=.8)
    ax.plot(observed_pfeatures[-1, :, 0], observed_pfeatures[-1, :, 1], 'x', label='observed', alpha=.8)
    ax.plot(estimated_pfeatures[-1, :, 0], estimated_pfeatures[-1, :, 1], 'o', label='estimated', alpha=.4)

    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    np.set_printoptions(linewidth=500, suppress=True)
    matplotlib.rc('font', size=9)
    matplotlib.rc('legend', fontsize=9)

    run_position_estimation()
