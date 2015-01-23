import collections
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
    if y[2] > 0:
        return geometry.pr(y)
    else:
        return None


class FeatureObservation(object):
    def __init__(self, frame_id, track_id, position):
        self.frame_id = frame_id
        self.track_id = track_id
        self.position = position


def select_by_timestamp(data, timestamps, begin, end):
    begin_index = bisect.bisect_left(timestamps, begin)
    end_index = bisect.bisect_right(timestamps, end)
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


def renumber_tracks(features, landmarks=None, min_track_length=None):
    # Drop tracks that are too short
    if min_track_length is not None:
        track_lengths = collections.defaultdict(int)
        for f in features:
            track_lengths[f.track_id] += 1
        features = filter(lambda f: track_lengths[f.track_id] >= min_track_length, features)

    # Apply consistent renumbering
    track_ids = sorted(set(f.track_id for f in features))
    track_index_by_id = {track_id: index for index, track_id in enumerate(track_ids)}
    for f in features:
        f.track_id = track_index_by_id[f.track_id]

    # Return the final features
    if landmarks is not None:
        assert len(landmarks) > max(track_ids)
        landmarks = np.array([landmarks[i] for i in track_ids])
        return features, landmarks
    else:
        return features


class FirstOrderRotationCurve(object):
    def __init__(self, timestamps, orientations):
        self.timestamps = timestamps
        self.orientations = orientations

    def evaluate(self, t):
        return cayley.cayley_inv(interpolate_orientation(self.timestamps, self.orientations, t))


class PositionEstimate(object):
    def __init__(self, position_curve, gravity, accel_bias, landmarks):
        self.position_curve = position_curve
        self.gravity = gravity
        self.accel_bias = accel_bias
        self.landmarks = landmarks

    @property
    def size(self):
        return len(self.flatten())

    def flatten(self):
        return np.hstack((self.position_curve.controls.flatten(),
                          self.gravity,
                          self.accel_bias,
                          self.landmarks.flatten()))


def householder(x):
    assert len(x) == 3, 'x=%s' % x
    assert np.linalg.norm(x) > 1e-8
    a = (np.arange(3) == np.argmin(np.abs(x))).astype(float)
    u = utils.normalized(np.cross(x, a))
    v = utils.normalized(np.cross(x, u))
    return np.array([u, v])


def calibrated(z, k):
    return utils.normalized(np.linalg.solve(k, utils.unpr(z)))


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
    # Sanity checks
    assert isinstance(spline_template, spline.SplineTemplate)
    assert len(observed_accel_orientations) == len(observed_accel_readings)
    assert len(observed_accel_timestamps) == len(observed_accel_readings)
    assert len(observed_frame_timestamps) == len(observed_frame_orientations)
    assert all(0 <= f.frame_id < len(observed_frame_timestamps) for f in observed_features)
    assert np.ndim(observed_accel_timestamps) == 1
    assert np.ndim(observed_frame_timestamps) == 1

    # Compute offsets
    position_offset = 0
    position_len = spline_template.control_size
    gravity_offset = position_offset + position_len
    accel_bias_offset = gravity_offset + 3
    structure_offset = accel_bias_offset + 3
    track_ids = set(f.track_id for f in observed_features)

    num_frames = len(observed_frame_timestamps)
    num_tracks = max(track_ids) + 1
    num_vars = structure_offset + num_tracks * 3

    # Make sure each track has at least one observation
    counts_by_frame = np.zeros(num_frames, int)
    counts_by_track = np.zeros(num_tracks, int)
    for f in observed_features:
        counts_by_frame[f.frame_id] += 1
        counts_by_track[f.track_id] += 1

    assert np.all(counts_by_frame > 0),\
        'These frames had zero features: ' + ','.join(map(str, np.flatnonzero(counts_by_frame == 0)))
    assert np.all(counts_by_track > 0),\
        'These tracks had zero features: ' + ','.join(map(str, np.flatnonzero(counts_by_track == 0)))


    # Track IDs should be exactly 0..n-1
    assert all(track_id < num_tracks for track_id in track_ids)

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
    print 'Constructing constraints for %d accel readings...' % len(observed_accel_readings)
    accel_coefficients = spline_template.coefficients_d2(observed_accel_timestamps)
    for r, a, c in zip(observed_accel_orientations, observed_accel_readings, accel_coefficients):
        amat = spline.diagify(c, 3)
        #amat = spline_template.multidim_coefficients_d2(t)
        j = np.zeros((3, num_vars))
        j[:, :position_len] = np.dot(r, amat)
        j[:, gravity_offset:gravity_offset+3] = r
        j[:, accel_bias_offset:accel_bias_offset+3] = np.eye(3)
        r = -a
        problem.add_constraint(a=j, b=r, d=accel_tolerance)

    # Construct vision constraints
    print 'Constructing constraints for %d features...' % len(observed_features)
    pos_coefficients = spline_template.coefficients(observed_frame_timestamps)
    pos_multidim_coefs = [spline.diagify(x, 3) for x in pos_coefficients]
    for feature in observed_features:
        r = observed_frame_orientations[feature.frame_id]
        pmat = pos_multidim_coefs[feature.frame_id]

        point_offset = structure_offset + feature.track_id*3
        assert point_offset + 3 <= num_vars, 'track id was %d, num vars was %d' % (feature.track_id, num_vars)

        k_rc_r = np.dot(camera_matrix, np.dot(imu_to_camera, r))
        ymat = np.zeros((3, num_vars))
        ymat[:, :position_len] = -np.dot(k_rc_r, pmat)
        ymat[:, point_offset:point_offset+3] = k_rc_r

        a_feature = ymat[:2] - np.outer(feature.position, ymat[2])
        c_feature = ymat[2] * feature_tolerance

        problem.add_constraint(a=a_feature, c=c_feature)

    return problem


def estimate_trajectory_socp(spline_template,
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
                             max_bias_magnitude=.1,
                             ground_truth=None):
    problem = construct_problem(
        spline_template,
        observed_accel_timestamps,
        observed_accel_orientations,
        observed_accel_readings,
        observed_frame_timestamps,
        observed_frame_orientations,
        observed_features,
        imu_to_camera=imu_to_camera,
        camera_matrix=camera_matrix,
        feature_tolerance=feature_tolerance,
        accel_tolerance=accel_tolerance,
        gravity_magnitude=gravity_magnitude,
        max_bias_magnitude=max_bias_magnitude)

    if ground_truth is not None:
        problem.evaluate(ground_truth.flatten())

    problem = problem.conditionalize_indices(range(3), np.zeros(3))
    result = socp.solve(problem, sparse=True)

    if result['x'] is None:
        print 'Did not find a feasible solution'
        return

    estimated_vars = np.hstack((np.zeros(3), np.squeeze(result['x'])))

    spline_vars = spline_template.control_size
    pos_controls = estimated_vars[:spline_vars].reshape((-1, 3))
    gravity = estimated_vars[spline_vars:spline_vars+3]
    accel_bias = estimated_vars[spline_vars+3:spline_vars+6]
    landmarks = estimated_vars[spline_vars+6:].reshape((-1, 3))

    curve = spline.Spline(spline_template, pos_controls)
    return PositionEstimate(curve, gravity, accel_bias, landmarks)


def estimate_trajectory_linear(spline_template,
                               observed_accel_timestamps,
                               observed_accel_orientations,
                               observed_accel_readings,
                               observed_frame_timestamps,
                               observed_frame_orientations,
                               observed_features,
                               imu_to_camera=np.eye(3),
                               camera_matrix=np.eye(3),
                               accel_weight=1.):
    assert isinstance(spline_template, spline.SplineTemplate)

    num_tracks = max(f.track_id for f in observed_features) + 1

    accel_bias_offset = spline_template.control_size
    gravity_offset = spline_template.control_size + 3
    structure_offset = spline_template.control_size + 6
    num_vars = structure_offset + num_tracks * 3

    j_blocks = []
    r_blocks = []

    # Add terms for accel residuals
    print 'Constructing linear systems for %d accel readings...' % len(observed_accel_readings)
    accel_coefficients = spline_template.coefficients_d2(observed_accel_timestamps)
    for r, a, c in zip(observed_accel_orientations, observed_accel_readings, accel_coefficients):
        amat = spline.diagify(c, 3)
        j = np.zeros((3, num_vars))
        j[:, :spline_template.control_size] = np.dot(r, amat)
        j[:, gravity_offset:gravity_offset+3] = r
        j[:, accel_bias_offset:accel_bias_offset+3] = np.eye(3)
        j_blocks.append(j * accel_weight)
        r_blocks.append(a * accel_weight)

    # Add terms for features
    print 'Constructing linear systems for %d features...' % len(observed_features)
    pos_coefficients = spline_template.coefficients(observed_frame_timestamps)
    pos_multidim_coefs = [spline.diagify(x, 3) for x in pos_coefficients]
    for feature in observed_features:
        z = calibrated(feature.position, camera_matrix)
        h = householder(z)
        r = observed_frame_orientations[feature.frame_id]
        pmat = pos_multidim_coefs[feature.frame_id]

        point_offset = structure_offset + feature.track_id*3
        j = np.zeros((2, num_vars))
        j[:, :spline_template.control_size] = -np.dot(h, np.dot(imu_to_camera, np.dot(r, pmat)))
        j[:, point_offset:point_offset+3] = np.dot(h, np.dot(imu_to_camera, r))

        j_blocks.append(j)
        r_blocks.append(np.zeros(2))

    # Assemble full linear system
    j = np.vstack(j_blocks)
    r = np.hstack(r_blocks)

    # Eliminate global position
    j = j[:, 3:]

    # Solve
    print 'Solving linear system of size %d x %d' % j.shape
    solution, _, _, _ = np.linalg.lstsq(j, r)

    # Replace global position
    solution = np.hstack((np.zeros(3), solution))

    # Extract individual variables from solution
    position_controls = solution[:spline_template.control_size].reshape((-1, 3))
    position_curve = spline.Spline(spline_template, position_controls)
    gravity = solution[gravity_offset:gravity_offset+3]
    accel_bias = solution[accel_bias_offset:accel_bias_offset+3]
    landmarks = solution[structure_offset:].reshape((-1, 3))
    return PositionEstimate(position_curve, gravity, accel_bias, landmarks)


def run_in_simulation():
    np.random.seed(0)

    #
    # Construct ground truth
    #
    duration = 20.
    num_frames = 100
    num_landmarks = 500
    num_imu_readings = 100

    degree = 3
    num_controls = 50
    num_knots = num_controls - degree + 1
    
    accel_timestamp_noise = 0
    accel_reading_noise = 1e-3
    accel_orientation_noise = 0

    frame_timestamp_noise = 0
    frame_orientation_noise = 0
    feature_noise = 1.

    spline_template = spline.SplineTemplate(np.linspace(0, duration, num_knots), degree, 3)

    print 'Num landmarks:', num_landmarks
    print 'Num frames:', num_frames
    print 'Num IMU readings:', num_imu_readings
    print 'Spline curve degree:', degree

    # Both splines should start at 0,0,0
    true_frame_timestamps = np.linspace(0, duration, num_frames)
    true_accel_timestamps = np.linspace(0, duration, num_imu_readings)

    true_rot_curve = spline_template.build_random(.1)
    true_pos_curve = spline_template.build_random(first_control=np.zeros(3))

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

    # Sample IMU readings
    true_imu_orientations = np.array(map(cayley.cayley, true_rot_curve.evaluate(true_accel_timestamps)))
    true_accel_readings = np.array([predict_accel(true_pos_curve, true_rot_curve, true_accel_bias, true_gravity, t)
                                    for t in true_accel_timestamps])

    # Sample features
    num_behind = 0
    true_features = []
    for frame_id, t in enumerate(true_frame_timestamps):
        r = cayley.cayley(true_rot_curve.evaluate(t))
        p = true_pos_curve.evaluate(t)
        a = np.dot(true_camera_matrix, np.dot(true_imu_to_camera, r))
        ys = np.dot(true_landmarks - p, a.T)
        for track_id, y in enumerate(ys):
            # predict_feature will return None if the landmark is behind the camera
            if y[2] > 0:
                true_features.append(FeatureObservation(frame_id, track_id, geometry.pr(y)))
            else:
                print 'omitting feature for frame %d, track %d' % (frame_id, track_id)
                num_behind += 1

    if num_behind > 0:
        print '%d landmarks were behind the camera (and %d were in front)' % (num_behind, len(true_features))

    true_features, true_landmarks = renumber_tracks(true_features, true_landmarks, min_track_length=2)
    true_trajectory = PositionEstimate(true_pos_curve, true_gravity, true_accel_bias, true_landmarks)

    #
    # Add sensor noise
    #
    observed_accel_timestamps = utils.add_white_noise(true_accel_timestamps, accel_timestamp_noise)
    observed_accel_readings = utils.add_white_noise(true_accel_readings, accel_reading_noise)
    observed_accel_orientations = utils.add_orientation_noise(true_imu_orientations, accel_orientation_noise)

    observed_frame_timestamps = utils.add_white_noise(true_frame_timestamps, frame_timestamp_noise)
    observed_frame_orientations = utils.add_orientation_noise(true_frame_orientations, frame_orientation_noise)

    observed_features = []
    for f in true_features:
        observed_features.append(FeatureObservation(f.frame_id,
                                                    f.track_id,
                                                    utils.add_white_noise(f.position, feature_noise)))

    #
    # Solve
    #
    solver = 'linear'
    if solver == 'socp':
        trajectory = estimate_trajectory_socp(spline_template,
                                              observed_accel_timestamps,
                                              observed_accel_orientations,
                                              observed_accel_readings,
                                              observed_frame_timestamps,
                                              observed_frame_orientations,
                                              observed_features,
                                              imu_to_camera=true_imu_to_camera,
                                              camera_matrix=true_camera_matrix,
                                              gravity_magnitude=true_gravity_magnitude+.1,
                                              accel_tolerance=1e-3,
                                              feature_tolerance=1.,
                                              ground_truth=true_trajectory)
    elif solver == 'linear':
        trajectory = estimate_trajectory_linear(spline_template,
                                                observed_accel_timestamps,
                                                observed_accel_orientations,
                                                observed_accel_readings,
                                                observed_frame_timestamps,
                                                observed_frame_orientations,
                                                observed_features,
                                                imu_to_camera=true_imu_to_camera,
                                                camera_matrix=true_camera_matrix)
    else:
        print 'Invalid solver:', solver
        return

    #
    # Visualize
    #
    estimated_frame_positions = trajectory.position_curve.evaluate(true_frame_timestamps)

    print 'Position errors:', np.linalg.norm(estimated_frame_positions - true_frame_positions, axis=1)
    print 'Gravity error:', np.linalg.norm(trajectory.gravity - true_gravity)
    print 'Accel bias error:', np.linalg.norm(trajectory.accel_bias - true_accel_bias)
    print 'Max error:', np.max(trajectory.flatten() - true_trajectory.flatten())

    # Plot the variables
    plt.clf()
    plt.barh(np.arange(true_trajectory.size), true_trajectory.flatten(), height=.3, alpha=.3, color='g')
    plt.barh(np.arange(true_trajectory.size)+.4, trajectory.flatten(), height=.3, alpha=.3, color='r')
    plt.savefig('out/vars.pdf')

    plot_timestamps = np.linspace(0, duration, 500)
    estimated_ps = trajectory.position_curve.evaluate(plot_timestamps)
    true_ps = true_pos_curve.evaluate(plot_timestamps)

    # Plot the estimated trajectory
    plt.clf()
    plt.plot(estimated_ps[:, 0], estimated_ps[:, 1], 'r-')
    plt.plot(true_ps[:, 0], true_ps[:, 1], 'b-')
    plt.axis('equal')
    plt.savefig('out/trajectory.pdf')


def run_with_dataset():
    dataset_path = '/tmp/dataset'
    vfusion_path = '/tmp/out'

    gravity = np.array([0, 0, 9.82])
    min_track_length = 6
    num_knots = 200

    # Load vision model
    vision_model = list(open(dataset_path + '/vision_model.txt'))
    camera_matrix = np.array(map(float, vision_model[0].split())).reshape((3, 3))
    imu_to_camera = np.array(map(float, vision_model[1].split())).reshape((3, 3))

    # Load frame timestamps
    all_frame_timestamps = np.loadtxt(dataset_path + '/frame_timestamps.txt')

    # Load accel data
    all_accel = np.loadtxt(dataset_path + '/accel.txt')

    # Load features
    all_features = []
    with open(dataset_path + '/features.txt') as fd:
        for line in fd:
            frame_id, track_id, x, y = line.split()
            all_features.append(FeatureObservation(int(frame_id), int(track_id), np.array([float(x), float(y)])))

    # Load trajectory from vfusion
    all_vfusion_states = np.loadtxt(vfusion_path + '/states.txt')
    all_vfusion_timestamps = all_vfusion_states[:, 1]

    begin_timestamp = all_vfusion_timestamps[0] + 10.
    end_timestamp = all_vfusion_timestamps[0] + 20.  #25.

    vfusion_states = select_by_timestamp(all_vfusion_states,
                                         all_vfusion_timestamps,
                                         begin_timestamp,
                                         end_timestamp)

    vfusion_timestamps = vfusion_states[:, 1]
    vfusion_orientations = vfusion_states[:, 2:11].reshape((-1, 3, 3))
    vfusion_positions = vfusion_states[:, -3:]
    vfusion_pos_curve = spline.fit(vfusion_timestamps, vfusion_positions, knot_frequency=1)
    vfusion_orientation_curve = FirstOrderRotationCurve(vfusion_timestamps, vfusion_orientations)

    vfusion_gyro_bias = vfusion_states[:, 11:14]
    vfusion_velocities = vfusion_states[:, 14:17]
    vfusion_accel_bias = vfusion_states[:, 17:20]

    print 'Max accel bias:', np.max(np.linalg.norm(vfusion_accel_bias, axis=1))

    # Set up IMU measurements
    accel = select_by_timestamp(all_accel, all_accel[:, 0], begin_timestamp, end_timestamp)
    accel_timestamps = accel[:, 0]
    accel_readings = accel[:, 1:]
    accel_orientations = [interpolate_orientation(vfusion_timestamps, vfusion_orientations, t)
                          for t in accel_timestamps]

    # Set up frames
    begin_frame_index = bisect.bisect_left(all_frame_timestamps, begin_timestamp)
    end_frame_index = bisect.bisect_left(all_frame_timestamps, end_timestamp)
    frame_timestamps = all_frame_timestamps[begin_frame_index:end_frame_index]
    frame_orientations = [interpolate_orientation(vfusion_timestamps, vfusion_orientations, t)
                          for t in frame_timestamps]

    # Set up features
    print 'Selecting frame indices %d...%d' % (begin_frame_index, end_frame_index)
    features = []
    track_lengths = collections.defaultdict(int)
    for f in all_features:
        if begin_frame_index <= f.frame_id < end_frame_index:
            features.append(f)
            track_lengths[f.track_id] += 1

    # Filter by track length
    features = filter(lambda f: track_lengths[f.track_id] >= min_track_length, features)
    print '  selected %d of %d features' % (len(features), len(all_features))

    # Renumber track IDs and frame_ids consecutively
    frame_ids = sorted(set(f.frame_id for f in features))
    track_ids = sorted(set(f.track_id for f in features))
    frame_index_by_id = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    track_index_by_id = {track_id: index for index, track_id in enumerate(track_ids)}
    for f in features:
        f.track_id = track_index_by_id[f.track_id]
        f.frame_id = frame_index_by_id[f.frame_id]

    # Create the problem
    print 'Creating problem for %d frames, %d tracks, %d features, and %d accel readings...' % \
          (len(frame_timestamps), len(track_ids), len(features), len(accel_readings))

    spline_tpl = spline.SplineTemplate.linspaced(num_knots, dims=3, begin=begin_timestamp, end=end_timestamp)
    problem = construct_problem(spline_tpl,
                                accel_timestamps,
                                accel_orientations,
                                accel_readings,
                                frame_timestamps,
                                frame_orientations,
                                features,
                                camera_matrix=camera_matrix,
                                imu_to_camera=imu_to_camera,
                                feature_tolerance=10.,
                                accel_tolerance=2.,
                                max_bias_magnitude=1.,
                                gravity_magnitude=np.linalg.norm(gravity) + .1)

    print 'Constructed a problem with %d variables and %d constraints' % \
          (len(problem.objective), len(problem.constraints))

    # Eliminate global position
    print 'Eliminating the first position...'
    problem = problem.conditionalize_indices(range(3), np.zeros(3))

    # Run solver
    print 'Solving...'
    result = socp.solve(problem, sparse=True)

    if result['x'] is None:
        print 'Did not find a feasible solution'
        return
    else:
        print 'Found a solution'

    # Unpack results
    estimated_vars = np.hstack((np.zeros(3), np.squeeze(result['x'])))

    spline_vars = spline_tpl.control_size
    estimated_pos_controls = estimated_vars[:spline_vars].reshape((-1, 3))
    estimated_accel_bias = estimated_vars[spline_vars:spline_vars+3]
    estimated_gravity = estimated_vars[spline_vars+3:spline_vars+6]
    estimated_landmarks = estimated_vars[spline_vars+6:].reshape((-1, 3))

    estimated_pos_curve = spline.Spline(spline_tpl, estimated_pos_controls)
    estimated_frame_positions = estimated_pos_curve.evaluate(frame_timestamps)
    vfusion_frame_positions = vfusion_pos_curve.evaluate(frame_timestamps)

    print 'Estimated gravity:', estimated_gravity
    print 'Estimated accel bias:', estimated_accel_bias
    print '  vfusion accel bias box: ', np.min(vfusion_accel_bias, axis=0), np.max(vfusion_accel_bias, axis=0)
    print 'Position errors:', np.linalg.norm(estimated_frame_positions - vfusion_frame_positions, axis=1)

    # Save results to file
    np.savetxt('/tmp/solution/estimated_frame_positions.txt', estimated_frame_positions)
    np.savetxt('/tmp/solution/estimated_pos_controls.txt', estimated_pos_controls)
    np.savetxt('/tmp/solution/estimated_accel_bias.txt', estimated_accel_bias)
    np.savetxt('/tmp/solution/estimated_gravity.txt', estimated_gravity)
    np.savetxt('/tmp/solution/estimated_landmarks.txt', estimated_landmarks)
    np.savetxt('/tmp/knots.txt', spline_tpl.knots)

    plot_timestamps = np.linspace(begin_timestamp, end_timestamp, 500)
    estimated_ps = estimated_pos_curve.evaluate(plot_timestamps)
    vfusion_ps = vfusion_pos_curve.evaluate(plot_timestamps)

    # Plot the estimated trajectory
    plt.clf()
    plt.plot(estimated_ps[:, 0], estimated_ps[:, 1], 'r-')
    plt.plot(vfusion_ps[:, 0], vfusion_ps[:, 1], 'b-')
    plt.axis('equal')
    plt.savefig('out/trajectory.pdf')

    # Plot the estimated trajectory at its own scale
    plt.clf()
    plt.plot(estimated_ps[:, 0], estimated_ps[:, 1], 'r-')
    plt.plot(estimated_pos_controls[:, 0], estimated_pos_controls[:, 1], 'b-', alpha=.2)
    plt.axis('equal')
    plt.savefig('out/lone_trajectory.pdf')

    # Plot the estimated vars
    plt.clf()
    plt.barh(np.arange(len(estimated_vars)), estimated_vars, height=.75, color='r')
    plt.savefig('out/vars.pdf')

    # Synthesize accel readings and compare to measured values
    timestamps = np.linspace(begin_timestamp, end_timestamp, 100)
    predicted_accels = []
    for t in timestamps:
        predicted_accels.append(predict_accel(estimated_pos_curve,
                                              vfusion_orientation_curve,
                                              estimated_accel_bias,
                                              estimated_gravity,
                                              t))

    predicted_accels = np.array(predicted_accels)

    plt.clf()
    plt.plot(timestamps, predicted_accels, '-', label='predicted')
    plt.plot(accel_timestamps, accel_readings, '-', label='observed')
    plt.legend()
    plt.savefig('out/accel.pdf')

    return

    # Plot
    ts = np.linspace(begin_timestamp, end_timestamp, 200)
    ys = vfusion_pos_curve.evaluate(ts)

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
