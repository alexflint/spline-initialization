import bisect
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import utils
import cayley
import spline
import socp
import geometry
import lie


def predict_accel(pos_curve, orient_curve, accel_bias, gravity, t):
    orientation = cayley.cayley(orient_curve.evaluate(t))
    global_accel = pos_curve.evaluate_d2(t)
    return np.dot(orientation, global_accel + gravity) + accel_bias


def predict_feature(pos_curve, orient_curve, landmark, t, imu_to_camera, camera_matrix):
    r = cayley.cayley(orient_curve.evaluate(t))
    p = pos_curve.evaluate(t)
    y = np.dot(camera_matrix, np.dot(imu_to_camera, np.dot(r, landmark - p)))
    assert y[2] > 0
    return geometry.pr(y)


class FeatureObservation(object):
    def __init__(self, frame_id, track_id, position):
        self.frame_id = frame_id
        self.track_id = track_id
        self.position = position


def features_from_positions(positions):
    """Given a NUMFRAMES x NUMTRACKS x 2 array, create a list of feature observations."""
    return [FeatureObservation(i, j, p) for i, row in enumerate(positions) for j, p in enumerate(row)]


def construct_problem(spline_template,
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
    assert isinstance(spline_template, spline.SplineTemplate)

    # Compute offsets
    position_offset = 0
    position_len = spline_template.control_size
    accel_bias_offset = position_offset + position_len
    gravity_offset = accel_bias_offset + 3
    structure_offset = gravity_offset + 3
    track_ids = set(f.track_id for f in observed_features)
    num_vars = structure_offset + 3 * len(track_ids)

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
        amat = spline_template.multidim_coefficients_d2(t)
        j = np.zeros((3, num_vars))
        j[:, :position_len] = np.dot(r, amat)
        j[:, gravity_offset:gravity_offset+3] = r
        j[:, accel_bias_offset:accel_bias_offset+3] = np.eye(3)
        r = -a
        problem.add_constraint(a=j, b=r, d=accel_tolerance)

    # Construct structure constraints
    for feature in observed_features:
        t = observed_frame_timestamps[feature.frame_id]
        r = observed_frame_orientations[feature.frame_id]

        point_offset = structure_offset + feature.track_id*3

        pmat = spline_template.multidim_coefficients(t)
        k_rc_r = np.dot(camera_matrix, np.dot(imu_to_camera, r))
        ymat = np.zeros((3, num_vars))
        ymat[:, :position_len] = -np.dot(k_rc_r, pmat)
        ymat[:, point_offset:point_offset+3] = k_rc_r

        a_feature = ymat[:2] - np.outer(feature.position, ymat[2])
        c_feature = ymat[2] * feature_tolerance

        problem.add_constraint(a=a_feature, c=c_feature)

    return problem


def run_in_simulation():
    np.random.seed(0)

    #
    # Construct ground truth
    #
    duration = 1.
    num_frames = 6
    num_landmarks = 10
    num_imu_readings = 100

    degree = 3
    num_controls = 6
    num_knots = num_controls - degree + 1
    
    spline_tpl = spline.SplineTemplate(np.linspace(0, duration, num_knots), degree, 3)

    print 'Num landmarks:', num_landmarks
    print 'Num frames:', num_frames
    print 'Num IMU readings:', num_imu_readings
    print 'Spline curve degree:', degree

    # Both splines should start at 0,0,0
    true_frame_timestamps = np.linspace(0, .9, num_frames)
    true_accel_timestamps = np.linspace(0, 1, num_imu_readings)

    true_rot_curve = spline_tpl.build_random(.1)
    true_pos_curve = spline_tpl.build_random(first_control=np.zeros(3))

    true_landmarks = np.random.randn(num_landmarks, 3)*5 + np.array([0., 0., 20.])

    true_frame_orientations = np.array(map(cayley.cayley, true_rot_curve.evaluate(true_frame_timestamps)))
    true_frame_positions = np.array(true_pos_curve.evaluate(true_frame_timestamps))

    true_gravity_magnitude = 9.8
    true_gravity = utils.normalized(np.random.rand(3)) * true_gravity_magnitude
    true_accel_bias = np.random.randn(3) * .01

    true_imu_to_camera = lie.SO3.exp(np.random.randn(3))
    true_camera_matrix = np.array([[150., 0., 75.],
                                   [0., 100., 50.],
                                   [0., 0., 1.]])

    print 'True gravity:', true_gravity

    true_imu_orientations = np.array(map(cayley.cayley, true_rot_curve.evaluate(true_accel_timestamps)))
    true_accel_readings = np.array([predict_accel(true_pos_curve, true_rot_curve, true_accel_bias, true_gravity, t)
                                    for t in true_accel_timestamps])

    true_features = np.array([[predict_feature(true_pos_curve, true_rot_curve, x, t, true_imu_to_camera, true_camera_matrix)
                               for x in true_landmarks]
                              for t in true_frame_timestamps])

    true_vars = np.hstack((true_pos_curve.controls.flatten(),
                           true_accel_bias,
                           true_gravity,
                           true_landmarks.flatten()))

    #
    # Add sensor noise
    #
    accel_timestamp_noise = 0
    accel_reading_noise = 0
    accel_orientation_noise = 0

    frame_timestamp_noise = 0
    frame_orientation_noise = 0
    feature_noise = 0

    observed_accel_timestamps = utils.add_white_noise(true_accel_timestamps, accel_timestamp_noise)
    observed_accel_readings = utils.add_white_noise(true_accel_readings, accel_reading_noise)
    observed_accel_orientations = utils.add_orientation_noise(true_imu_orientations, accel_orientation_noise)

    observed_frame_timestamps = utils.add_white_noise(true_frame_timestamps, frame_timestamp_noise)
    observed_frame_orientations = utils.add_orientation_noise(true_frame_orientations, frame_orientation_noise)
    observed_feature_positions = utils.add_white_noise(true_features, feature_noise)
    observed_features = features_from_positions(observed_feature_positions)

    #
    # Solve
    #
    problem = construct_problem(
        spline_tpl,
        observed_accel_timestamps,
        observed_accel_orientations,
        observed_accel_readings,
        observed_frame_timestamps,
        observed_frame_orientations,
        observed_features,
        gravity_magnitude=true_gravity_magnitude+.1,
        accel_tolerance=1e-3,
        feature_tolerance=1.,
        imu_to_camera=true_imu_to_camera,
        camera_matrix=true_camera_matrix)

    problem = problem.conditionalize_indices(range(3), np.zeros(3))
    result = socp.solve(problem, sparse=True)

    if result['x'] is None:
        print 'Did not find a feasible solution'
        return

    estimated_vars = np.hstack((np.zeros(3), np.squeeze(result['x'])))

    spline_vars = spline_tpl.control_size
    estimated_pos_controls = estimated_vars[:spline_vars].reshape((-1, 3))
    estimated_accel_bias = estimated_vars[spline_vars:spline_vars+3]
    estimated_gravity = estimated_vars[spline_vars+3:spline_vars+6]
    estimated_landmarks = estimated_vars[spline_vars+6:].reshape((-1, 3))

    estimated_pos_curve = spline.Spline(spline_tpl, estimated_pos_controls)
    estimated_frame_positions = estimated_pos_curve.evaluate(true_frame_timestamps)

    print 'Position norms:', np.linalg.norm(true_frame_positions, axis=1)
    print 'Position errors:', np.linalg.norm(estimated_frame_positions - true_frame_positions, axis=1)

    print 'Gravity error:', np.linalg.norm(estimated_gravity - true_gravity)
    print 'Accel bias error:', np.linalg.norm(estimated_accel_bias - true_accel_bias)

    print 'Max error:', np.max(estimated_vars - true_vars)

    plt.clf()
    plt.barh(np.arange(len(true_vars)), true_vars, height=.3, alpha=.3, color='g')
    plt.barh(np.arange(len(true_vars))+.4, estimated_vars, height=.3, alpha=.3, color='r')
    plt.savefig('out/vars.pdf')


def select_by_timestamp(data, timestamps, begin, end):
    begin_index = bisect.bisect(timestamps, begin)
    end_index = bisect.bisect(timestamps, end)
    return data[begin_index:end_index]


def interpolate(timestamps, data, t):
    assert len(timestamps) == len(data)
    index = bisect.bisect_left(timestamps, t)
    if index == 0:
        return data[0]
    elif index == len(data):
        return data[-1]
    else:
        x0, x1 = data[index-1:index+1]
        t0, t1 = timestamps[index-1:index+1]
        a = (t - t0) / (t1 - t0)
        return (1.-a)*x0 + a*x1


def interpolate_orientation(timestamps, orientations, t):
    assert len(timestamps) == len(orientations)
    index = bisect.bisect_left(timestamps, t)
    if index == 0:
        return orientations[0]
    elif index == len(orientations):
        return orientations[-1]
    else:
        r0, r1 = orientations[index-1:index+1]
        t0, t1 = timestamps[index-1:index+1]
        diff = lie.SO3.log(np.dot(r0.T, r1))
        w = diff * (t - t0) / (t1 - t0)
        return np.dot(r0, lie.SO3.exp(w))


class FirstOrderRotationCurve(object):
    def __init__(self, timestamps, orientations):
        self.timestamps = timestamps
        self.orientations = orientations

    def evaluate(self, t):
        return cayley.cayley_inv(interpolate_orientation(self.timestamps, self.orientations, t))


def run_with_dataset():
    dataset_path = '/tmp/dataset'
    vfusion_path = '/tmp/out'

    # Load raw data
    all_accel = np.loadtxt(dataset_path + '/accel.txt')

    vision_model = list(open(dataset_path + '/vision_model.txt'))
    camera_matrix = np.array(map(float, vision_model[0].split())).reshape((3, 3))
    imu_to_camera = np.array(map(float, vision_model[1].split())).reshape((3, 3))

    # Load trajectory from vfusion
    all_vfusion_states = np.loadtxt(vfusion_path + '/states.txt')
    all_vfusion_timestamps = all_vfusion_states[:, 1]

    begin_timestamp = all_vfusion_timestamps[0] + 10.
    end_timestamp = all_vfusion_timestamps[0] + 25.

    vfusion_states = select_by_timestamp(all_vfusion_states,
                                         all_vfusion_timestamps,
                                         begin_timestamp,
                                         end_timestamp)

    vfusion_timestamps = vfusion_states[:, 1]
    vfusion_orientations = vfusion_states[:, 2:11].reshape((-1, 3, 3))
    vfusion_positions = vfusion_states[:, -3:]
    vfusion_curve = spline.fit(vfusion_timestamps, vfusion_positions, knot_frequency=1)
    orientation_curve = FirstOrderRotationCurve(vfusion_timestamps, vfusion_orientations)

    vfusion_gyro_bias = vfusion_states[:, 11:14]
    vfusion_velocities = vfusion_states[:, 14:17]
    vfusion_accel_bias = vfusion_states[:, 17:20]

    # Predict one accel reading
    accel = select_by_timestamp(all_accel, all_accel[:, 0], begin_timestamp, end_timestamp)

    predicted_accels = []
    observed_accels = []
    timestamps = np.linspace(begin_timestamp, end_timestamp, 100)
    for t in vfusion_timestamps:  #timestamps:
        ba = interpolate(vfusion_timestamps, vfusion_accel_bias, t)
        gravity = np.array([0, 0, 9.82])

        predicted_accel = predict_accel(vfusion_curve, orientation_curve, ba, gravity, t)
        observed_accel = interpolate(accel[:, 0], accel[:, 1:], t)

        print '\nPredicted:', predicted_accel
        print 'Observed:', observed_accel
        print 'Diff:', predicted_accel - observed_accel

        predicted_accels.append(predicted_accel)
        observed_accels.append(observed_accel)

    predicted_accels = np.array(predicted_accels)
    observed_accels = np.array(observed_accels)

    plt.clf()
    plt.plot(timestamps, predicted_accels[:, 0], label='predicted')
    plt.plot(timestamps, observed_accels[:, 0], label='observed')
    plt.plot(accel[:, 0], accel[:, 1], '-', alpha=.2)
    plt.legend()
    plt.savefig('out/accel.pdf')

    return

    # Plot
    ts = np.linspace(begin_timestamp, end_timestamp, 200)
    ys = vfusion_curve.evaluate(ts)

    plt.clf()
    plt.plot(vfusion_positions[:, 0], vfusion_positions[:, 1], 'gx', alpha=.5)
    plt.plot(ys[:, 0], ys[:, 1], 'r-', alpha=.4)
    plt.axis('equal')
    plt.savefig('out/vfusion_positions.pdf')


def run_fit_spline():
    ts = np.linspace(0, 10, 10)
    ys = np.random.randn(len(ts))
    curve = spline.fit(ts, ys, num_knots=8)

    plt.clf()
    plot_ts = np.linspace(0, 10, 200)
    plt.plot(plot_ts, curve.evaluate(plot_ts), 'r-')
    plt.plot(ts, ys, 'xk')
    plt.savefig('out/fit.pdf')


def run_fit_spline_multidim():
    ts = np.linspace(0, 10, 9)
    ys = np.random.randn(len(ts), 3)
    curve = spline.fit(ts, ys, num_knots=8)

    plt.clf()
    plot_ts = np.linspace(0, 10, 200)
    plot_ys = curve.evaluate(plot_ts)
    plt.plot(plot_ys[:, 0], plot_ys[:, 1], 'r-')
    plt.plot(ys[:, 0], ys[:, 1], 'xk')
    plt.savefig('out/fit.pdf')


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)
    run_in_simulation()
    #run_with_dataset()
    #run_fit_spline()
    #run_fit_spline_multidim()
