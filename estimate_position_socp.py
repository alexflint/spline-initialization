import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import utils
import cayley
import bezier
import socp
import geometry


def predict_accel(pos_controls, orient_controls, accel_bias, gravity, t):
    orientation = cayley.cayley(bezier.zero_offset_bezier(orient_controls, t))
    global_accel = bezier.zero_offset_bezier_second_deriv(pos_controls, t)
    return np.dot(orientation, global_accel + gravity) + accel_bias


def predict_feature(pos_controls, orient_controls, landmark, t):
    r = cayley.cayley(bezier.zero_offset_bezier(orient_controls, t))
    p = bezier.zero_offset_bezier(pos_controls, t)
    y = np.dot(r, landmark - p)
    assert y[2] > 0
    return geometry.pr(y)


def predict_depth(pos_controls, orient_controls, landmark, t):
    r = cayley.cayley(bezier.zero_offset_bezier(orient_controls, t))
    p = bezier.zero_offset_bezier(pos_controls, t)
    return np.linalg.norm(np.dot(r, landmark - p))


def construct_problem(bezier_degree,
                      observed_accel_timestamps,
                      observed_accel_orientations,
                      observed_accel_readings,
                      observed_frame_timestamps,
                      observed_frame_orientations,
                      observed_features,
                      imu_to_camera=np.eye(3),
                      camera_matrix=np.eye(3),
                      feature_tolerance=1e-2,
                      accel_tolerance=1e-3,
                      gravity_magnitude=9.8,
                      max_bias_magnitude=.1):
    # Compute offsets
    position_offset = 0
    position_len = (bezier_degree-1)*3
    accel_bias_offset = position_offset + position_len
    gravity_offset = accel_bias_offset + 3
    structure_offset = gravity_offset + 3
    num_vars = structure_offset + 3 * observed_features.shape[1]

    # Initialize the problem
    objective = np.zeros(num_vars)
    problem = socp.SocpProblem(objective, [])

    # Construct gravity constraints
    a_gravity = np.zeros((3, num_vars))
    a_gravity[:, gravity_offset:gravity_offset+3] = np.eye(3)
    d_gravity = gravity_magnitude
    problem.add_constraint(a=a_gravity, d=d_gravity)

    # Construct accel bias constraints
    a_bias = np.zeros((3, num_vars))
    a_bias[:, accel_bias_offset:accel_bias_offset+3] = np.eye(3)
    d_bias = max_bias_magnitude
    problem.add_constraint(a=a_bias, d=d_bias)

    # Construct accel constraints
    for t, r, a in zip(observed_accel_timestamps, observed_accel_orientations, observed_accel_readings):
        amat = bezier.zero_offset_bezier_second_deriv_mat(t, bezier_degree-1, 3)
        j = np.zeros((3, num_vars))
        j[:, :position_len] = np.dot(r, amat)
        j[:, gravity_offset:gravity_offset+3] = r
        j[:, accel_bias_offset:accel_bias_offset+3] = np.eye(3)
        r = -a
        problem.add_constraint(a=j, b=r, d=accel_tolerance)

    # Construct structure constraints
    for i, (t, r, zs) in enumerate(zip(observed_frame_timestamps, observed_frame_orientations, observed_features)):
        for j, z in enumerate(zs):
            point_offset = structure_offset + j*3

            pmat = bezier.zero_offset_bezier_mat(t, bezier_degree-1, 3)
            k_rc_r = np.dot(camera_matrix, np.dot(imu_to_camera, r))
            ymat = np.zeros((3, num_vars))
            ymat[:, :position_len] = -np.dot(k_rc_r, pmat)
            ymat[:, point_offset:point_offset+3] = k_rc_r

            a_feature = ymat[:2] - np.outer(z, ymat[2])
            c_feature = ymat[2] * feature_tolerance

            problem.add_constraint(a=a_feature, c=c_feature)

    return problem


def run_position_estimation():
    np.random.seed(0)

    #
    # Construct ground truth
    #
    num_frames = 6
    num_landmarks = 50
    num_imu_readings = 100
    bezier_degree = 4

    print 'Num landmarks:', num_landmarks
    print 'Num frames:', num_frames
    print 'Num IMU readings:', num_imu_readings
    print 'Bezier curve degree:', bezier_degree

    # Both splines should start at 0,0,0
    true_frame_timestamps = np.linspace(0, .9, num_frames)
    true_accel_timestamps = np.linspace(0, 1, num_imu_readings)

    true_rot_controls = np.random.randn(bezier_degree-1, 3) * .1
    true_pos_controls = np.random.randn(bezier_degree-1, 3)

    true_landmarks = np.random.randn(num_landmarks, 3)*5 + [0., 0., 20.]

    true_frame_orientations = np.array([cayley.cayley(bezier.zero_offset_bezier(true_rot_controls, t))
                                        for t in true_frame_timestamps])
    true_frame_positions = np.array([bezier.zero_offset_bezier(true_pos_controls, t) for t in true_frame_timestamps])

    true_gravity_magnitude = 9.8
    true_gravity = utils.normalized(np.random.rand(3)) * true_gravity_magnitude
    true_accel_bias = np.random.randn(3) * .01

    print 'True gravity:', true_gravity

    true_imu_orientations = np.array([cayley.cayley(bezier.zero_offset_bezier(true_rot_controls, t))
                                      for t in true_accel_timestamps])
    true_accel_readings = np.array([predict_accel(true_pos_controls, true_rot_controls, true_accel_bias, true_gravity, t)
                                    for t in true_accel_timestamps])

    true_features = np.array([[predict_feature(true_pos_controls, true_rot_controls, x, t) for x in true_landmarks]
                              for t in true_frame_timestamps])

    true_vars = np.hstack((true_pos_controls.flatten(), true_accel_bias, true_gravity, true_landmarks.flatten()))

    #
    # Add sensor noise
    #
    accel_timestamp_noise = 0.
    accel_reading_noise = 0.
    accel_orientation_noise = 0.

    frame_timestamp_noise = 0
    frame_orientation_noise = 0.
    feature_noise = 0.

    observed_accel_timestamps = utils.add_white_noise(true_accel_timestamps, accel_timestamp_noise)
    observed_accel_readings = utils.add_white_noise(true_accel_readings, accel_reading_noise)
    observed_accel_orientations = utils.add_orientation_noise(true_imu_orientations, accel_orientation_noise)

    observed_frame_timestamps = utils.add_white_noise(true_frame_timestamps, frame_timestamp_noise)
    observed_frame_orientations = utils.add_orientation_noise(true_frame_orientations, frame_orientation_noise)
    observed_features = utils.add_white_noise(true_features, feature_noise)

    #
    # Solve
    #
    problem = construct_problem(
        bezier_degree,
        observed_accel_timestamps,
        observed_accel_orientations,
        observed_accel_readings,
        observed_frame_timestamps,
        observed_frame_orientations,
        observed_features,
        gravity_magnitude=true_gravity_magnitude+.1,
        accel_tolerance=1e-2,
        feature_tolerance=1e-2)

    problem.evaluate(true_vars)

    begin = time.clock()
    result = socp.solve(problem)
    duration = time.clock() - begin

    print 'Solve took %.3fs' % duration

    estimated_vars = np.squeeze(result['x'])

    print 'estimated:', estimated_vars.shape
    print 'true:', true_vars.shape

    print 'Max error:', np.max(estimated_vars - true_vars)
    print '  at ', np.argmax(estimated_vars - true_vars)

    plt.clf()
    plt.barh(np.arange(len(true_vars)), true_vars, height=.3, alpha=.3, color='g')
    plt.barh(np.arange(len(true_vars))+.4, estimated_vars, height=.3, alpha=.3, color='r')
    plt.savefig('out/vars.pdf')


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)
    run_position_estimation()
