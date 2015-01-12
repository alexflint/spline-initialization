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
from plotting import plot_tracks


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


def reprojection_residual(pos_controls, landmark, depth, timestamp, feature, orientation):
    position = zero_offset_bezier(pos_controls, timestamp)
    return np.dot(orientation, landmark - position) - depth * feature


def reprojection_jacobian(bezier_order, timestamp, feature, orientation):
    bezier_mat = zero_offset_bezier_mat(timestamp, bezier_order, 3)
    return -np.dot(orientation, bezier_mat), orientation, -feature


def evaluate_reprojection_residuals(pos_controls, landmarks, depths, frame_timestamps, frame_orientations,
                                    features, feature_mask=None):
    residuals = []
    for i, (t, R) in enumerate(zip(frame_timestamps, frame_orientations)):
        for k, (z, landmark, depth) in enumerate(zip(features[i], landmarks, depths[i])):
            if feature_mask is None or feature_mask[i, k]:
                residuals.append(reprojection_residual(pos_controls, landmark, depth, t, z, R))
    return np.hstack(residuals)


def evaluate_reprojection_jacobians(bezier_order, frame_timestamps, frame_orientations,
                                    features, feature_mask=None):
    num_frames, num_landmarks = np.shape(features)[:2]
    position_offset = 0
    accel_bias_offset = position_offset + bezier_order*3
    gravity_offset = accel_bias_offset + 3
    landmark_offset = gravity_offset + 3
    depth_offset = landmark_offset + num_landmarks * 3
    size = depth_offset + num_frames * num_landmarks

    jacobians = []
    for i, (t, R) in enumerate(zip(frame_timestamps, frame_orientations)):
        for j, z in enumerate(features[i]):
            if feature_mask is None or feature_mask[i, j]:
                J_r_wrt_p, J_r_wrt_x, J_r_wrt_k = reprojection_jacobian(bezier_order, t, z, R)
                x_offset = landmark_offset + j * 3
                k_offset = depth_offset + i * num_landmarks + j
                jcur = np.zeros((3, size))
                jcur[:, position_offset:accel_bias_offset] = J_r_wrt_p
                jcur[:, x_offset:x_offset+3] = J_r_wrt_x
                jcur[:, k_offset] = J_r_wrt_k
                jacobians.append(jcur)
    return np.vstack(jacobians)


def ba_reprojection_residual(pos_controls, landmark, timestamp, feature, orientation):
    position = zero_offset_bezier(pos_controls, timestamp)
    return pr(np.dot(orientation, landmark - position)) - pr(feature)


def evaluate_ba_reprojection_residuals(pos_controls, landmarks, frame_timestamps, frame_orientations,
                                       features, feature_mask=None):
    residuals = []
    for i, (t, R) in enumerate(zip(frame_timestamps, frame_orientations)):
        for k, (z, x) in enumerate(zip(features[i], landmarks)):
            if feature_mask is None or feature_mask[i, k]:
                residuals.append(ba_reprojection_residual(pos_controls, x, t, z, R))
    return np.hstack(residuals)


def estimate_position(bezier_degree,
                      observed_accel_timestamps,
                      observed_accel_orientations,
                      observed_accel_readings,
                      observed_frame_timestamps,
                      observed_frame_orientations,
                      observed_features,
                      vision_weight=1.):
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

    residual = np.hstack((accel_res, epipolar_res * vision_weight))
    jacobian = np.vstack((accel_jac, epipolar_jac * vision_weight))

    # Solve
    jtj = np.dot(jacobian.T, jacobian)
    jtr = np.dot(jacobian.T, residual)
    estimated_vars = np.squeeze(np.linalg.solve(jtj, -jtr))

    # Unpack result and compute error
    estimated_pos_controls = estimated_vars[position_offs:position_offs+(bezier_degree-1)*3].reshape((bezier_degree-1, 3))
    estimated_accel_bias = estimated_vars[accel_bias_offset:accel_bias_offset+3]
    estimated_gravity = estimated_vars[gravity_offset:gravity_offset+3]
    return estimated_pos_controls, estimated_accel_bias, estimated_gravity


def structure_and_motion_system(bezier_degree,
                                observed_accel_timestamps,
                                observed_accel_orientations,
                                observed_accel_readings,
                                observed_frame_timestamps,
                                observed_frame_orientations,
                                observed_features,
                                vision_weight=1.):
    # Compute offsets for parameter vector
    num_frames, num_landmarks = observed_features.shape[:2]

    linearization_accel_bias = np.zeros(3)
    linearization_gravity = np.zeros(3)
    #linearization_gravity = np.array([3.01579968, 8.26799292, 4.31106082])
    linearization_pos_controls = np.zeros((bezier_degree-1, 3))
    linearization_landmarks = np.zeros((num_landmarks, 3))
    linearization_depths = np.zeros((num_frames, num_landmarks))

    # Setup linear system
    accel_res = evaluate_accel_residuals(linearization_pos_controls,
                                         linearization_accel_bias,
                                         linearization_gravity,
                                         observed_accel_timestamps,
                                         observed_accel_readings,
                                         observed_accel_orientations)
    accel_jac = evaluate_accel_jacobians(bezier_degree-1,
                                         observed_accel_timestamps,
                                         observed_accel_orientations)
    accel_jac = np.hstack((accel_jac, np.zeros((accel_jac.shape[0], num_landmarks * (3 + num_frames)))))

    reproj_res = evaluate_reprojection_residuals(linearization_pos_controls,
                                                 linearization_landmarks,
                                                 linearization_depths,
                                                 observed_frame_timestamps,
                                                 observed_frame_orientations,
                                                 observed_features)
    reproj_jac = evaluate_reprojection_jacobians(bezier_degree-1,
                                                 observed_frame_timestamps,
                                                 observed_frame_orientations,
                                                 observed_features)

    residual = np.hstack((accel_res, reproj_res * vision_weight))
    jacobian = np.vstack((accel_jac, reproj_jac * vision_weight))
    return residual, jacobian


def estimate_structure_and_motion(bezier_degree,
                                  observed_accel_timestamps,
                                  observed_accel_orientations,
                                  observed_accel_readings,
                                  observed_frame_timestamps,
                                  observed_frame_orientations,
                                  observed_features,
                                  vision_weight=1.):
    # Compute offsets for parameter vector
    num_frames, num_landmarks = observed_features.shape[:2]
    position_offset = 0
    accel_bias_offset = position_offset + (bezier_degree-1)*3
    gravity_offset = accel_bias_offset + 3
    landmark_offset = gravity_offset + 3
    depth_offset = landmark_offset + num_landmarks * 3

    residual, jacobian = structure_and_motion_system(bezier_degree,
                                                     observed_accel_timestamps,
                                                     observed_accel_orientations,
                                                     observed_accel_readings,
                                                     observed_frame_timestamps,
                                                     observed_frame_orientations,
                                                     observed_features,
                                                     vision_weight)
    # Solve
    jtj = np.dot(jacobian.T, jacobian)
    jtr = np.dot(jacobian.T, residual)

    #jtj[gravity_offset:landmark_offset, :] = 0
    #jtj[:, gravity_offset:landmark_offset] = 0
    #jtj[gravity_offset:landmark_offset, gravity_offset:landmark_offset] = np.eye(3)
    #jtr[gravity_offset:landmark_offset] = 0

    estimated_vars = np.squeeze(np.linalg.solve(jtj, -jtr))

    print 'Linear system error:', np.linalg.norm(np.dot(jtj, estimated_vars) + jtr)

    # Unpack result and compute error
    estimated_pos_controls = estimated_vars[position_offset:accel_bias_offset].reshape((-1, 3))
    estimated_accel_bias = estimated_vars[accel_bias_offset:gravity_offset]
    estimated_gravity = estimated_vars[gravity_offset:landmark_offset]
    estimated_landmarks = estimated_vars[landmark_offset:depth_offset].reshape((-1, 3))
    estimated_depths = estimated_vars[depth_offset:].reshape((num_frames, num_landmarks))
    return estimated_pos_controls, estimated_accel_bias, estimated_gravity, estimated_landmarks, estimated_depths


def optimize_ba(bezier_degree,
                init_pos_controls,
                init_accel_bias,
                init_gravity,
                init_landmarks,
                observed_accel_timestamps,
                observed_accel_orientations,
                observed_accel_readings,
                observed_frame_timestamps,
                observed_frame_orientations,
                observed_features,
                vision_weight=1.):
    # Compute offsets for parameter vector
    num_frames, num_landmarks = observed_features.shape[:2]
    position_offset = 0
    accel_bias_offset = position_offset + (bezier_degree-1)*3
    gravity_offset = accel_bias_offset + 3
    landmark_offset = gravity_offset + 3

    def unpack(x):
        pos_controls = x[position_offset:accel_bias_offset].reshape((-1, 3))
        accel_bias = x[accel_bias_offset:gravity_offset]
        gravity = x[gravity_offset:landmark_offset]
        landmarks = x[landmark_offset:].reshape((-1, 3))
        return pos_controls, accel_bias, gravity, landmarks

    def residual(x):
        pos_controls, accel_bias, gravity, landmarks = unpack(x)
        reprojection_residual = evaluate_ba_reprojection_residuals(pos_controls,
                                                                   landmarks,
                                                                   observed_frame_timestamps,
                                                                   observed_frame_orientations,
                                                                   observed_features)
        accel_residual = evaluate_accel_residuals(pos_controls,
                                                  accel_bias,
                                                  gravity,
                                                  observed_accel_timestamps,
                                                  observed_accel_readings,
                                                  observed_accel_orientations)
        return np.hstack((reprojection_residual, accel_residual))

    residual, jacobian = structure_and_motion_system(bezier_degree,
                                                     observed_accel_timestamps,
                                                     observed_accel_orientations,
                                                     observed_accel_readings,
                                                     observed_frame_timestamps,
                                                     observed_frame_orientations,
                                                     observed_features,
                                                     vision_weight)
    # Solve
    jtj = np.dot(jacobian.T, jacobian)
    jtr = np.dot(jacobian.T, residual)

    #jtj[gravity_offset:landmark_offset, :] = 0
    #jtj[:, gravity_offset:landmark_offset] = 0
    #jtj[gravity_offset:landmark_offset, gravity_offset:landmark_offset] = np.eye(3)
    #jtr[gravity_offset:landmark_offset] = 0

    estimated_vars = np.squeeze(np.linalg.solve(jtj, -jtr))

    print 'Linear system error:', np.linalg.norm(np.dot(jtj, estimated_vars) + jtr)

    # Unpack result and compute error
    estimated_pos_controls = estimated_vars[position_offset:accel_bias_offset].reshape((-1, 3))
    estimated_accel_bias = estimated_vars[accel_bias_offset:gravity_offset]
    estimated_gravity = estimated_vars[gravity_offset:landmark_offset]
    estimated_landmarks = estimated_vars[landmark_offset:depth_offset].reshape((-1, 3))
    estimated_depths = estimated_vars[depth_offset:].reshape((num_frames, num_landmarks))
    return estimated_pos_controls, estimated_accel_bias, estimated_gravity, estimated_landmarks, estimated_depths


def predict_accel(pos_controls, orient_controls, accel_bias, gravity, t):
    orientation = cayley(zero_offset_bezier(orient_controls, t))
    global_accel = zero_offset_bezier_second_deriv(pos_controls, t)
    return np.dot(orientation, global_accel + gravity) + accel_bias


def predict_feature(pos_controls, orient_controls, landmark, t):
    r = cayley(zero_offset_bezier(orient_controls, t))
    p = zero_offset_bezier(pos_controls, t)
    return normalized(np.dot(r, landmark - p))


def predict_depth(pos_controls, orient_controls, landmark, t):
    r = cayley(zero_offset_bezier(orient_controls, t))
    p = zero_offset_bezier(pos_controls, t)
    return np.linalg.norm(np.dot(r, landmark - p))


def run_position_estimation():
    #
    # Construct ground truth
    #
    num_frames = 5
    num_landmarks = 150
    num_imu_readings = 80
    bezier_degree = 4
    use_epipolar_constraints = False

    print 'Num landmarks:', num_landmarks
    print 'Num frames:', num_frames
    print 'Num IMU readings:', num_imu_readings
    print 'Bezier curve degree:', bezier_degree

    # Both splines should start at 0,0,0
    true_frame_timestamps = np.linspace(0, .9, num_frames)
    true_accel_timestamps = np.linspace(0, 1, num_imu_readings)

    true_rot_controls = np.random.randn(bezier_degree-1, 3)
    true_pos_controls = np.random.randn(bezier_degree-1, 3)

    true_landmarks = np.random.randn(num_landmarks, 3) * 5
    true_landmarks[:, 2] += 20

    true_frame_orientations = np.array([cayley(zero_offset_bezier(true_rot_controls, t)) for t in true_frame_timestamps])
    true_frame_positions = np.array([zero_offset_bezier(true_pos_controls, t) for t in true_frame_timestamps])

    true_gravity_magnitude = 9.8
    true_gravity = normalized(np.random.rand(3)) * true_gravity_magnitude
    true_accel_bias = np.random.randn(3)

    print 'True gravity:', true_gravity

    true_imu_orientations = np.array([cayley(zero_offset_bezier(true_rot_controls, t)) for t in true_accel_timestamps])
    true_accel_readings = np.array([predict_accel(true_pos_controls, true_rot_controls, true_accel_bias, true_gravity, t)
                                    for t in true_accel_timestamps])

    true_features = np.array([[predict_feature(true_pos_controls, true_rot_controls, x, t) for x in true_landmarks]
                              for t in true_frame_timestamps])

    true_depths = np.array([[predict_depth(true_pos_controls, true_rot_controls, x, t) for x in true_landmarks]
                            for t in true_frame_timestamps])

    #
    # Add sensor noise
    #

    accel_timestamp_noise = 0
    accel_reading_noise = 1e-3
    accel_orientation_noise = 1e-3

    frame_timestamp_noise = 0
    frame_orientation_noise = 1e-3
    feature_noise = 5e-3

    observed_accel_timestamps = add_white_noise(true_accel_timestamps, accel_timestamp_noise)
    observed_accel_readings = add_white_noise(true_accel_readings, accel_reading_noise)
    observed_accel_orientations = add_orientation_noise(true_imu_orientations, accel_orientation_noise)

    observed_frame_timestamps = add_white_noise(true_frame_timestamps, frame_timestamp_noise)
    observed_frame_orientations = add_orientation_noise(true_frame_orientations, frame_orientation_noise)
    observed_features = add_white_noise(true_features, feature_noise)

    #
    # Plot
    #
    #plt.clf()
    #plot_tracks(true_features, 'g-', alpha=.2, limit=1)
    #plot_tracks(true_features, 'go', alpha=.6, limit=1)
    #plot_tracks(observed_features, 'r-', alpha=.2, limit=1)
    #plot_tracks(observed_features, 'rx', alpha=.6, limit=1)
    #plt.show()

    #
    # Solve
    #

    if use_epipolar_constraints:
        estimated_pos_controls, estimated_accel_bias, estimated_gravity = estimate_position(
            bezier_degree,
            observed_accel_timestamps,
            observed_accel_orientations,
            observed_accel_readings,
            observed_frame_timestamps,
            observed_frame_orientations,
            observed_features,
            vision_weight=1.)
    else:
        estimated_pos_controls, estimated_accel_bias, estimated_gravity, estimated_landmarks, estimated_depths = \
            estimate_structure_and_motion(
                bezier_degree,
                observed_accel_timestamps,
                observed_accel_orientations,
                observed_accel_readings,
                observed_frame_timestamps,
                observed_frame_orientations,
                observed_features,
                vision_weight=1.)

        r, j = structure_and_motion_system(
            bezier_degree,
            observed_accel_timestamps,
            observed_accel_orientations,
            observed_accel_readings,
            observed_frame_timestamps,
            observed_frame_orientations,
            observed_features,
            vision_weight=1.)

        t0 = observed_frame_timestamps[0]
        r0 = observed_frame_orientations[0]
        z0 = observed_features[0, 0]
        p0 = zero_offset_bezier(estimated_pos_controls, t0)
        pp0 = zero_offset_bezier(true_pos_controls, t0)
        x0 = estimated_landmarks[0]
        xx0 = true_landmarks[0]
        k0 = estimated_depths[0, 0]
        kk0 = np.linalg.norm(np.dot(r0, xx0 - pp0))
        print 'residual:'
        print reprojection_residual(estimated_pos_controls, x0, k0, t0, z0, r0)
        print reprojection_residual(true_pos_controls, xx0, kk0, t0, z0, r0)
        print np.dot(r0, x0 - p0) - k0 * z0
        print np.dot(r0, xx0 - pp0) - kk0 * z0

        #true_structure = np.hstack((true_landmarks, true_depths[:, None]))
        #true_params = np.hstack((true_pos_controls.flatten(), true_accel_bias, true_gravity, true_structure.flatten()))
        #jtj = np.dot(j.T, j)
        #jtr = np.dot(j.T, r)
        #print jtj.shape, true_params.shape, jtr.shape
        #print np.dot(jtj, true_params) - jtr

        #return

    estimated_positions = np.array([zero_offset_bezier(estimated_pos_controls, t)
                                    for t in true_frame_timestamps])

    estimated_accel_readings = np.array([predict_accel(estimated_pos_controls,
                                                       true_rot_controls,
                                                       estimated_accel_bias,
                                                       estimated_gravity,
                                                       t)
                                         for t in true_accel_timestamps])

    estimated_pfeatures = np.array([[pr(predict_feature(estimated_pos_controls, true_rot_controls, x, t))
                                     for x in true_landmarks]
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


def run_reprojection_finite_differences():
    np.random.seed(0)

    bezier_order = 4
    pos_controls = np.random.randn(bezier_order, 3)
    landmark = np.random.randn(3)
    depth = np.random.randn()
    timestamp = .5
    feature = np.random.randn(3)
    orientation = SO3.exp(np.random.randn(3))

    position_offset = 0
    accel_bias_offset = position_offset + bezier_order * 3
    gravity_offset = accel_bias_offset + 3
    landmark_offset = gravity_offset + 3
    depth_offset = landmark_offset + 3
    size = depth_offset + 1

    def r(delta):
        return reprojection_residual(
            pos_controls + delta[:accel_bias_offset].reshape(pos_controls.shape),
            landmark + delta[landmark_offset:depth_offset],
            depth + delta[depth_offset],
            timestamp,
            feature,
            orientation)

    J_numeric = numdifftools.Jacobian(r)(np.zeros(size))

    J_wrt_p, J_wrt_x, J_wrt_k = reprojection_jacobian(bezier_order, timestamp, feature, orientation)
    print J_wrt_p.shape, J_wrt_x.shape, J_wrt_k[:,None].shape
    J_analytic = np.hstack((J_wrt_p, np.zeros((3, 6)), J_wrt_x, J_wrt_k[:, None]))

    print '\nNumeric:'
    print J_numeric

    print '\nAnalytic:'
    print J_analytic

    np.testing.assert_array_almost_equal(J_numeric, J_analytic, decimal=8)


if __name__ == '__main__':
    np.random.seed(1)
    np.set_printoptions(linewidth=500, suppress=True)
    matplotlib.rc('font', size=9)
    matplotlib.rc('legend', fontsize=9)

    #run_reprojection_finite_differences()
    run_position_estimation()
