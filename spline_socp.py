import collections
import bisect
import numpy as np

import matplotlib
matplotlib.use('Agg', warn=False)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

import utils
import cayley
import spline
import socp
import geometry
import lie
import plotting
import triangulation


def predict_accel(pos_curve, orient_curve, accel_bias, gravity, t):
    orientation = cayley.cayley(orient_curve.evaluate(t))
    return predict_accel_with_orientation(pos_curve, orientation, accel_bias, gravity, t)


def predict_accel_with_orientation(pos_curve, orientation, accel_bias, gravity, t):
    global_accel = pos_curve.evaluate_d2(t)
    return np.dot(orientation, global_accel + gravity) + accel_bias


def predict_feature(pos_curve, orient_curve, landmark, t, imu_to_camera, camera_matrix):
    p = pos_curve.evaluate(t)
    r = cayley.cayley(orient_curve.evaluate(t))
    return predict_feature_with_pose(r, p, landmark, imu_to_camera, camera_matrix)


def predict_feature_with_pose(r, p, x, imu_to_camera, camera_matrix, allow_behind=True):
    assert np.shape(r) == (3, 3), 'shape was '+str(np.shape(r))
    assert np.shape(p) == (3,), 'shape was '+str(np.shape(p))
    assert np.shape(x) == (3,), 'shape was '+str(np.shape(x))
    assert np.shape(imu_to_camera) == (3, 3), 'shape was '+str(np.shape(imu_to_camera))
    assert np.shape(camera_matrix) == (3, 3), 'shape was '+str(np.shape(camera_matrix))
    y = np.dot(camera_matrix, np.dot(imu_to_camera, np.dot(r, x - p)))
    if not allow_behind and y[2] <= 0:
        return None
    return geometry.pr(y)


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
        assert len(track_ids) == 0 or len(landmarks) > max(track_ids)
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


def soc_constraint_from_quadratic_constraint(a, b, c):
    """Convert a quadratic constraint of the form

        x' A' A x + b' x + c <= 0

    to an equivalent SOCP constraint of the form

        || Q x + r ||_2 <= s' x + t
    """
    q = np.vstack((b/2., a))
    r = np.hstack((c/2. + .5, np.zeros(len(a))))
    s = -b/2.
    t = -c/2. + .5
    return q, r, s, t


def construct_problem_inf(spline_template,
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


def compute_accel_residuals(trajectory,
                            timestamps,
                            orientations,
                            readings):
    assert len(orientations) == len(readings)
    assert len(timestamps) == len(readings)
    assert np.ndim(timestamps) == 1

    residuals = []
    for t, r, a in zip(timestamps, orientations, readings):
        prediction = predict_accel_with_orientation(trajectory.position_curve,
                                                    r,
                                                    trajectory.accel_bias,
                                                    trajectory.gravity,
                                                    t)
        residuals.append(prediction - a)
    return np.hstack(residuals)


def compute_reprojection_errors(features, frame_timestamps, frame_orientations, estimated,
                                imu_to_camera, camera_matrix):
    errors = []
    frame_positions = estimated.position_curve.evaluate(frame_timestamps)
    for feature in features:
        r = frame_orientations[feature.frame_id]
        p = frame_positions[feature.frame_id]
        x = estimated.landmarks[feature.track_id]
        z = predict_feature_with_pose(r, p, x, imu_to_camera, camera_matrix)
        errors.append(z - feature.position)
    return np.array(errors)


def plot_features(features, frame_timestamps, frame_orientations, estimated, imu_to_camera, camera_matrix, output):
    """Synthesize features for each frame and compare to observations."""
    features_by_frame = [[] for _ in frame_timestamps]
    predictions_by_frame = [[] for _ in frame_timestamps]
    predicted_frame_positions = estimated.position_curve.evaluate(frame_timestamps)
    num_behind = 0
    for feature in features:
        r = frame_orientations[feature.frame_id]
        p = predicted_frame_positions[feature.frame_id]
        x = estimated.landmarks[feature.track_id]
        z = predict_feature_with_pose(r, p, x, imu_to_camera, camera_matrix)
        if z is None:
            num_behind += 1
        else:
            predictions_by_frame[feature.frame_id].append(z)
            features_by_frame[feature.frame_id].append(feature.position)

    if num_behind > 0:
        print '%d features (of %d) were behind the camera' % (num_behind, len(features))

    xmin, _, xmax = utils.minmedmax([f.position[0] for f in features])
    ymin, _, ymax = utils.minmedmax([f.position[1] for f in features])

    pdf = PdfPages(output)
    for i, (zs, zzs) in enumerate(zip(predictions_by_frame, features_by_frame)):
        print 'Plotting %d features for frame %d...' % (len(zs), i)
        zs = np.asarray(zs)
        zzs = np.asarray(zzs)
        plt.clf()
        if len(zs) > 0:
            plotting.plot_segments(zip(zs, zzs), '.-k', alpha=.5)
            plt.plot(zs[:, 0], zs[:, 1], '.r', alpha=.8)
            plt.plot(zzs[:, 0], zzs[:, 1], '.g', alpha=.8)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        pdf.savefig()
    pdf.close()


class InsufficientObservationsError(Exception):
    pass


def construct_problem_mixed(spline_template,
                            observed_accel_timestamps,
                            observed_accel_orientations,
                            observed_accel_readings,
                            observed_frame_timestamps,
                            observed_frame_orientations,
                            observed_features,
                            imu_to_camera=np.eye(3),
                            camera_matrix=np.eye(3),
                            feature_tolerance=1e-2,
                            gravity_magnitude=9.8,
                            max_bias_magnitude=.1):
    # Sanity checks
    assert len(observed_features) > 0
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

    num_aux_vars = 1  # one extra variable representing the objective
    num_frames = len(observed_frame_timestamps)
    num_tracks = max(track_ids) + 1
    num_vars = structure_offset + num_tracks * 3 + num_aux_vars

    # Make sure each track has at least one observation
    counts_by_frame = np.zeros(num_frames, int)
    counts_by_track = np.zeros(num_tracks, int)
    for f in observed_features:
        counts_by_frame[f.frame_id] += 1
        counts_by_track[f.track_id] += 1

    if not np.all(counts_by_frame > 0):
        raise InsufficientObservationsError(
            'These frames had zero features: ' + ','.join(map(str, np.flatnonzero(counts_by_frame == 0))))
    if not np.all(counts_by_track > 0):
        raise InsufficientObservationsError(
            'These tracks had zero features: ' + ','.join(map(str, np.flatnonzero(counts_by_track == 0))))

    # Track IDs should be exactly 0..n-1
    assert all(track_id < num_tracks for track_id in track_ids)

    # Initialize the problem
    objective = utils.unit(num_vars-1, num_vars)   # the last variable is the objective we minimize
    problem = socp.SocpProblem(objective)

    # Construct accel constraints
    print 'Constructing constraints for %d accel readings...' % len(observed_accel_readings)
    accel_coefficients = spline_template.coefficients_d2(observed_accel_timestamps)
    accel_j_blocks = []
    accel_r_blocks = []
    for r, a, c in zip(observed_accel_orientations, observed_accel_readings, accel_coefficients):
        amat = spline.diagify(c, 3)
        j = np.zeros((3, num_vars))
        j[:, :position_len] = np.dot(r, amat)
        j[:, gravity_offset:gravity_offset+3] = r
        j[:, accel_bias_offset:accel_bias_offset+3] = np.eye(3)
        accel_j_blocks.append(j)
        accel_r_blocks.append(a)

    # Form the least squares objective || J*x + r ||^2
    accel_j = np.vstack(accel_j_blocks)
    accel_r = np.hstack(accel_r_blocks)

    # Form the quadratic objective: x' J' J x + b' x + c <= objective  ("objective" is the variable we minimize)
    accel_c = np.dot(accel_r, accel_r)
    accel_b = -2. * np.dot(accel_j.T, accel_r)
    accel_b[-1] = -1.

    # Convert to an SOCP objective
    problem.add_constraint(*soc_constraint_from_quadratic_constraint(accel_j, accel_b, accel_c))

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


def estimate_trajectory_inf(spline_template,
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
                             ground_truth=None,
                             **kwargs):
    problem = construct_problem_inf(
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

    print 'Constructed a problem with %d variables and %d constraints' % \
          (len(problem.objective), len(problem.constraints))

    # Evaluate at ground truth if requested
    if ground_truth is not None:
        problem.evaluate(ground_truth.flatten())

    # Eliminate global position
    print 'Eliminating the first position...'
    problem = problem.conditionalize_indices(range(3))

    # Solve
    result = socp.solve(problem, sparse=True, **kwargs)

    if result['x'] is None:
        return None

    estimated_vars = np.hstack((np.zeros(3), np.squeeze(result['x'])))

    spline_vars = spline_template.control_size
    pos_controls = estimated_vars[:spline_vars].reshape((-1, 3))
    gravity = estimated_vars[spline_vars:spline_vars+3]
    accel_bias = estimated_vars[spline_vars+3:spline_vars+6]
    landmarks = estimated_vars[spline_vars+6:].reshape((-1, 3))

    curve = spline.Spline(spline_template, pos_controls)
    return PositionEstimate(curve, gravity, accel_bias, landmarks)


def estimate_trajectory_mixed(spline_template,
                              observed_accel_timestamps,
                              observed_accel_orientations,
                              observed_accel_readings,
                              observed_frame_timestamps,
                              observed_frame_orientations,
                              observed_features,
                              imu_to_camera=np.eye(3),
                              camera_matrix=np.eye(3),
                              feature_tolerance=1e-2,
                              gravity_magnitude=9.8,
                              max_bias_magnitude=.1,
                              ground_truth=None,
                              **kwargs):
    problem = construct_problem_mixed(
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
        gravity_magnitude=gravity_magnitude,
        max_bias_magnitude=max_bias_magnitude)

    print 'Constructed a problem with %d variables and %d constraints' % \
          (len(problem.objective), len(problem.constraints))

    # Evaluate at ground truth if requested
    if ground_truth is not None:
        # In the mixed formulation, the last variable is the sum of squared accel residuals
        gt_accel_residuals = compute_accel_residuals(ground_truth,
                                                     observed_accel_timestamps,
                                                     observed_accel_orientations,
                                                     observed_accel_readings)
        gt_cost = np.dot(gt_accel_residuals, gt_accel_residuals)
        ground_truth_augmented = np.hstack((ground_truth.flatten(), gt_cost * (1. + 1e-8)))
        problem.evaluate(ground_truth_augmented)

    # Eliminate global position
    print 'Eliminating the first position...'
    problem = problem.conditionalize_indices(range(3), np.zeros(3))

    # Solve
    result = socp.solve(problem, sparse=True, **kwargs)

    if result['x'] is None:
        return None

    estimated_vars = np.hstack((np.zeros(3), np.squeeze(result['x'])))

    spline_vars = spline_template.control_size
    pos_controls = estimated_vars[:spline_vars].reshape((-1, 3))
    gravity = estimated_vars[spline_vars:spline_vars+3]
    accel_bias = estimated_vars[spline_vars+3:spline_vars+6]
    landmarks = estimated_vars[spline_vars+6:-1].reshape((-1, 3))

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
        z = feature.position
        r = observed_frame_orientations[feature.frame_id]
        q = np.dot(camera_matrix, np.dot(imu_to_camera, r))
        c = q[:2] - np.outer(z, q[2])
        pmat = pos_multidim_coefs[feature.frame_id]

        point_offset = structure_offset + feature.track_id*3
        j = np.zeros((2, num_vars))
        j[:, :spline_template.control_size] = -np.dot(c, pmat)
        j[:, point_offset:point_offset+3] = c

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


def estimate_trajectory_householder(spline_template,
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


class Measurements(object):
    def __init__(self,
                 accel_timestamps,
                 accel_orientations,
                 accel_readings,
                 frame_timestamps,
                 frame_orientations,
                 features):
        self.accel_timestamps = accel_timestamps
        self.accel_orientations = accel_orientations
        self.accel_readings = accel_readings
        self.frame_timestamps = frame_timestamps
        self.frame_orientations = frame_orientations
        self.features = features


class Calibration(object):
    def __init__(self,
                 imu_to_camera,
                 camera_matrix,
                 gravity_magnitude):
        self.imu_to_camera = imu_to_camera
        self.camera_matrix = camera_matrix
        self.gravity_magnitude = gravity_magnitude

    @classmethod
    def random(cls, image_width=320., image_height=240., gravity_magnitude=9.8):
        imu_to_camera = lie.SO3.exp(np.random.randn(3))
        camera_matrix = np.array([[image_width, 0., image_width/2.],
                                  [0., image_height, image_height/2.],
                                  [0., 0., 1.]])
        return Calibration(imu_to_camera, camera_matrix, gravity_magnitude)


def simulate_trajectory(calibration,
                        duration=5.,
                        num_frames=12,
                        num_landmarks=50,
                        num_imu_readings=100,
                        degree=3,
                        num_controls=8,
                        accel_timestamp_noise=0.,
                        accel_reading_noise=0.,
                        accel_orientation_noise=0.,
                        frame_timestamp_noise=0.,
                        frame_orientation_noise=0.,
                        feature_noise=0.):
    num_knots = num_controls - degree + 1
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

    landmark_generator = 'normal'
    if landmark_generator == 'normal':
        true_landmarks = np.random.randn(num_landmarks, 3)*10
    elif landmark_generator == 'near':
        true_landmarks = []
        for i in range(num_landmarks):
            p = true_pos_curve.evaluate(true_frame_timestamps[i % len(true_frame_timestamps)])
            true_landmarks.append(p + np.random.randn()*.1)
    elif landmark_generator == 'far':
        true_landmarks = []
        for _ in range(num_landmarks):
            true_landmarks.append(utils.normalized(np.random.randn(3)) * 100000.)

    true_landmarks = np.asarray(true_landmarks)
    true_frame_orientations = np.array(map(cayley.cayley, true_rot_curve.evaluate(true_frame_timestamps)))

    true_gravity_magnitude = 9.8
    true_gravity = utils.normalized(np.random.rand(3)) * true_gravity_magnitude
    true_accel_bias = np.random.randn(3) * .01

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
        a = np.dot(calibration.camera_matrix, np.dot(calibration.imu_to_camera, r))
        ys = np.dot(true_landmarks - p, a.T)
        for track_id, y in enumerate(ys):
            if y[2] > 0:
                true_features.append(FeatureObservation(frame_id, track_id, geometry.pr(y)))
            else:
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

    if len(observed_features) < 5:
        raise InsufficientObservationsError()

    measurements = Measurements(observed_accel_timestamps,
                                observed_accel_orientations,
                                observed_accel_readings,
                                observed_frame_timestamps,
                                observed_frame_orientations,
                                observed_features)

    return true_trajectory, measurements, spline_template, true_frame_timestamps


def estimate_trajectory(calibration,
                        measurements,
                        spline_template,
                        estimator='mixed',
                        feature_tolerance=5.,
                        accel_tolerance=.1,
                        ground_truth=None):
    if estimator == 'socp':
        return estimate_trajectory_inf(spline_template,
                                       measurements.accel_timestamps,
                                       measurements.accel_orientations,
                                       measurements.accel_readings,
                                       measurements.frame_timestamps,
                                       measurements.frame_orientations,
                                       measurements.features,
                                       imu_to_camera=calibration.imu_to_camera,
                                       camera_matrix=calibration.camera_matrix,
                                       gravity_magnitude=calibration.gravity_magnitude+.1,
                                       feature_tolerance=feature_tolerance,
                                       accel_tolerance=accel_tolerance,
                                       ground_truth=ground_truth)
    elif estimator == 'mixed':
        return estimate_trajectory_mixed(spline_template,
                                         measurements.accel_timestamps,
                                         measurements.accel_orientations,
                                         measurements.accel_readings,
                                         measurements.frame_timestamps,
                                         measurements.frame_orientations,
                                         measurements.features,
                                         imu_to_camera=calibration.imu_to_camera,
                                         camera_matrix=calibration.camera_matrix,
                                         gravity_magnitude=calibration.gravity_magnitude+.1,
                                         feature_tolerance=feature_tolerance,
                                         ground_truth=ground_truth)
    elif estimator == 'householder':
        return estimate_trajectory_householder(spline_template,
                                               measurements.accel_timestamps,
                                               measurements.accel_orientations,
                                               measurements.accel_readings,
                                               measurements.frame_timestamps,
                                               measurements.frame_orientations,
                                               measurements.features,
                                               imu_to_camera=calibration.imu_to_camera,
                                               camera_matrix=calibration.camera_matrix)
    elif estimator == 'linear':
        return estimate_trajectory_linear(spline_template,
                                          measurements.accel_timestamps,
                                          measurements.accel_orientations,
                                          measurements.accel_readings,
                                          measurements.frame_timestamps,
                                          measurements.frame_orientations,
                                          measurements.features,
                                          imu_to_camera=calibration.imu_to_camera,
                                          camera_matrix=calibration.camera_matrix)
    else:
        print 'Invalid solver:', estimator
        return


def visualize_simulation_results(true_trajectory, estimated_trajectory, frame_timestamps):
    true_frame_positions = true_trajectory.position_curve.evaluate(frame_timestamps)
    estimated_frame_positions = estimated_trajectory.position_curve.evaluate(frame_timestamps)

    print 'Position errors:', np.linalg.norm(estimated_frame_positions - true_frame_positions, axis=1)
    print 'Gravity error:', np.linalg.norm(estimated_trajectory.gravity - true_trajectory.gravity)
    print 'Accel bias error:', np.linalg.norm(estimated_trajectory.accel_bias - true_trajectory.accel_bias)
    print 'Max error:', np.max(estimated_trajectory.flatten() - true_trajectory.flatten())

    # Plot the variables
    plt.clf()
    plt.barh(np.arange(true_trajectory.size), true_trajectory.flatten(), height=.3, alpha=.3, color='g')
    plt.barh(np.arange(estimated_trajectory.size)+.4, estimated_trajectory.flatten(), height=.3, alpha=.3, color='r')
    plt.savefig('out/vars.pdf')

    plot_timestamps = np.linspace(frame_timestamps[0], frame_timestamps[-1], 500)
    true_ps = true_trajectory.position_curve.evaluate(plot_timestamps)
    estimated_ps = estimated_trajectory.position_curve.evaluate(plot_timestamps)

    # Plot the estimated trajectory
    plt.clf()
    plt.plot(estimated_ps[:, 0], estimated_ps[:, 1], 'r-')
    plt.plot(true_ps[:, 0], true_ps[:, 1], 'b-')
    plt.axis('equal')
    plt.savefig('out/trajectory.pdf')


def mean_position_error(true_trajectory, estimated_trajectory, frame_timestamps):
    actual = true_trajectory.position_curve.evaluate(frame_timestamps)
    estimated = estimated_trajectory.position_curve.evaluate(frame_timestamps)
    return np.mean(np.linalg.norm(estimated - actual, axis=1))


def mean_velocity_error(true_trajectory, estimated_trajectory, frame_timestamps):
    actual = true_trajectory.position_curve.evaluate_d1(frame_timestamps)
    estimated = estimated_trajectory.position_curve.evaluate_d1(frame_timestamps)
    return np.mean(np.linalg.norm(estimated - actual, axis=1))


def gravity_direction_error(true_trajectory, estimated_trajectory):
    actual = utils.normalized(true_trajectory.gravity)
    estimated = utils.normalized(estimated_trajectory.gravity)
    return np.arccos(np.dot(actual, estimated))


def gravity_magnitude_error(true_trajectory, estimated_trajectory):
    actual = np.linalg.norm(true_trajectory.gravity)
    estimated = np.linalg.norm(estimated_trajectory.gravity)
    return np.abs(actual - estimated)


def accel_bias_error(true_trajectory, estimated_trajectory):
    actual = true_trajectory.accel_bias
    estimated = estimated_trajectory.accel_bias
    return np.linalg.norm(actual - estimated)


def run_simulation_series():
    np.random.seed(0)

    duration = 5.
    num_frames = 8
    num_landmarks = 50
    num_imu_readings = 100
    degree = 3
    num_controls = 8
    accel_timestamp_noise = 0
    accel_reading_noise = 1e-3
    accel_orientation_noise = 0
    frame_timestamp_noise = 0
    frame_orientation_noise = 0
    feature_noise = 1.

    num_trials = 1000

    calibration = Calibration.random()

    trials = []
    while len(trials) < num_trials:
        try:
            true_trajectory, measurements, spline_template, true_frame_timestamps = simulate_trajectory(
                calibration,
                duration=duration,
                num_frames=num_frames,
                num_landmarks=num_landmarks,
                num_imu_readings=num_imu_readings,
                degree=degree,
                num_controls=num_controls,
                accel_timestamp_noise=accel_timestamp_noise,
                accel_reading_noise=accel_reading_noise,
                accel_orientation_noise=accel_orientation_noise,
                frame_timestamp_noise=frame_timestamp_noise,
                frame_orientation_noise=frame_orientation_noise,
                feature_noise=feature_noise)
            row = []
            for estimator in ('socp', 'householder'):
                estimated_trajectory = estimate_trajectory(calibration,
                                                           measurements,
                                                           spline_template,
                                                           estimator=estimator,
                                                           feature_tolerance=feature_noise*3)
                pos_err = mean_position_error(true_trajectory, estimated_trajectory, true_frame_timestamps)
                vel_err = mean_velocity_error(true_trajectory, estimated_trajectory, true_frame_timestamps)
                bias_err = accel_bias_error(true_trajectory, estimated_trajectory)
                g_err = gravity_direction_error(true_trajectory, estimated_trajectory)
                row.extend((pos_err, vel_err, bias_err, g_err))
            trials.append(row)
        except InsufficientObservationsError:
            print 'Simulator failed to generate a reasonable trajectory. Retrying...'

    np.savetxt('results/trials.txt', trials)


def run_in_simulation():
    np.random.seed(1)
    true_trajectory, estimated_trajectory, true_frame_timestamps = simulate_and_estimate_trajectory()
    visualize_simulation_results(true_trajectory, estimated_trajectory, true_frame_timestamps)


def run_with_dataset():
    dataset_path = '/tmp/dataset'
    vfusion_path = '/tmp/out'

    gravity = np.array([0, 0, 9.82])
    min_track_length = 3
    max_frames = 100
    min_features_per_frame = 10
    max_iters = 100

    begin_time_offset = 5.
    end_time_offset = 7.
    knot_frequency = 10
    num_knots = int(np.ceil((end_time_offset - begin_time_offset) * knot_frequency))

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

    begin_timestamp = all_vfusion_timestamps[0] + begin_time_offset
    end_timestamp = all_vfusion_timestamps[0] + end_time_offset

    vfusion_states = select_by_timestamp(all_vfusion_states,
                                         all_vfusion_timestamps,
                                         begin_timestamp,
                                         end_timestamp)

    vfusion_timestamps = vfusion_states[:, 1]
    vfusion_orientations = vfusion_states[:, 2:11].reshape((-1, 3, 3))
    vfusion_positions = vfusion_states[:, -3:]
    vfusion_gyro_bias = vfusion_states[:, 11:14]
    vfusion_velocities = vfusion_states[:, 14:17]
    vfusion_accel_bias = vfusion_states[:, 17:20]

    vfusion_orientation_curve = FirstOrderRotationCurve(vfusion_timestamps, vfusion_orientations)
    vfusion_pos_curve = spline.fit(vfusion_timestamps, vfusion_positions, knot_frequency=1)

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
    if end_frame_index - begin_frame_index <= max_frames:
        selected_frame_ids = np.arange(begin_frame_index, end_frame_index, dtype=int)
    else:
        selected_frame_ids = np.linspace(begin_frame_index, end_frame_index-1, max_frames).round().astype(int)

    print 'Selected frames:', selected_frame_ids

    frame_timestamps = all_frame_timestamps[selected_frame_ids]
    frame_orientations = [interpolate_orientation(vfusion_timestamps, vfusion_orientations, t)
                          for t in frame_timestamps]
    frame_seed_positions = vfusion_pos_curve.evaluate(frame_timestamps)

    # Set up features
    print 'Selecting frame indices %d...%d' % (begin_frame_index, end_frame_index)
    tracks_by_id = collections.defaultdict(list)
    for f in all_features:
        if f.frame_id in selected_frame_ids:
            tracks_by_id[f.track_id].append(f)

    # Filter by track length
    tracks = filter(lambda t: len(t) >= min_track_length, tracks_by_id.viewvalues())
    track_counts = {index: 0 for index in selected_frame_ids}
    for track in tracks:
        for f in track:
            track_counts[f.frame_id] += 1

    # Filter tracks by track length, max tracks, and min features per frame
    features = []
    num_tracks_added = 0
    sorted_tracks = sorted(tracks, key=len)
    for track in sorted_tracks:
        if any(track_counts[f.frame_id] <= min_features_per_frame for f in track):
            num_tracks_added += 1
            features.extend(track)
        else:
            for f in track:
                track_counts[f.frame_id] -= 1

    print '  selected %d of %d features' % (len(features), len(all_features))
    print '  features per frame: ', ' '.join(map(str, track_counts.viewvalues()))

    # Renumber track IDs and frame_ids consecutively
    frame_ids = sorted(selected_frame_ids)
    track_ids = sorted(set(f.track_id for f in features))
    frame_index_by_id = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    track_index_by_id = {track_id: index for index, track_id in enumerate(track_ids)}
    for f in features:
        f.track_id = track_index_by_id[f.track_id]
        f.frame_id = frame_index_by_id[f.frame_id]

    # Create vfusion estimate
    tracks_by_id = collections.defaultdict(list)
    for f in features:
        tracks_by_id[f.track_id].append(f)
    vfusion_landmarks = np.array([triangulation.triangulate_midpoint(tracks_by_id[i],
                                                                     frame_orientations,
                                                                     frame_seed_positions,
                                                                     imu_to_camera,
                                                                     camera_matrix)
                                  for i in range(len(tracks_by_id))])
    print 'landmarks:'
    print vfusion_landmarks
    vfusion_estimate = PositionEstimate(vfusion_pos_curve, gravity, vfusion_accel_bias, vfusion_landmarks)

    vfusion_reproj_errors = compute_reprojection_errors(features, frame_timestamps, frame_orientations,
                                                        vfusion_estimate, imu_to_camera, camera_matrix)

    features = [f for f, err in zip(features, vfusion_reproj_errors) if np.linalg.norm(err) < 5.]
    features, vfusion_estimate.landmarks = renumber_tracks(features, vfusion_estimate.landmarks, min_track_length=2)

    # Plot the reprojected landmarks
    plot_features(features, frame_timestamps, frame_orientations, vfusion_estimate,
                  imu_to_camera, camera_matrix, 'out/vfusion_features.pdf')

    # Create the problem
    print 'Creating problem for %d frames, %d tracks, %d features, and %d accel readings...' % \
          (len(frame_timestamps), len(track_ids), len(features), len(accel_readings))

    spline_tpl = spline.SplineTemplate.linspaced(num_knots, dims=3, begin=begin_timestamp, end=end_timestamp)
    estimator = 'mixed'
    if estimator == 'socp':
        estimated = estimate_trajectory_inf(spline_tpl,
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
                                            gravity_magnitude=np.linalg.norm(gravity) + .1,
                                            maxiters=max_iters)
    elif estimator == 'mixed':
        estimated = estimate_trajectory_mixed(spline_tpl,
                                              accel_timestamps,
                                              accel_orientations,
                                              accel_readings,
                                              frame_timestamps,
                                              frame_orientations,
                                              features,
                                              camera_matrix=camera_matrix,
                                              imu_to_camera=imu_to_camera,
                                              feature_tolerance=2.,
                                              max_bias_magnitude=1.,
                                              gravity_magnitude=np.linalg.norm(gravity) + .1,
                                              maxiters=max_iters)
    elif estimator == 'linear':
        estimated = estimate_trajectory_linear(spline_tpl,
                                               accel_timestamps,
                                               accel_orientations,
                                               accel_readings,
                                               frame_timestamps,
                                               frame_orientations,
                                               features,
                                               camera_matrix=camera_matrix,
                                               imu_to_camera=imu_to_camera,
                                               accel_weight=100.)
    elif estimator == 'lsqnonlin':
        estimated = estimate_trajectory_lsqnonlin(spline_tpl,
                                                  accel_timestamps,
                                                  accel_orientations,
                                                  accel_readings,
                                                  frame_timestamps,
                                                  frame_orientations,
                                                  features,
                                                  camera_matrix=camera_matrix,
                                                  imu_to_camera=imu_to_camera,
                                                  accel_weight=100.,
                                                  seed=vfusion_estimate)
    else:
        print 'Invalid solver:', estimator
        return

    if estimated is None:
        print 'No solution found'
        return

    estimated_frame_positions = estimated.position_curve.evaluate(frame_timestamps)
    vfusion_frame_positions = vfusion_pos_curve.evaluate(frame_timestamps)

    print 'Estimated gravity:', estimated.gravity
    print 'Estimated accel bias:', estimated.accel_bias
    print '  vfusion accel bias box: ', np.min(vfusion_accel_bias, axis=0), np.max(vfusion_accel_bias, axis=0)
    print 'Position errors:', np.linalg.norm(estimated_frame_positions - vfusion_frame_positions, axis=1)

    # Save results to file
    np.savetxt('/tmp/solution/estimated_frame_positions.txt', estimated_frame_positions)
    np.savetxt('/tmp/solution/estimated_pos_controls.txt', estimated.position_curve.controls)
    np.savetxt('/tmp/solution/estimated_accel_bias.txt', estimated.accel_bias)
    np.savetxt('/tmp/solution/estimated_gravity.txt', estimated.gravity)
    np.savetxt('/tmp/solution/estimated_landmarks.txt', estimated.landmarks)
    np.savetxt('/tmp/solution/knots.txt', spline_tpl.knots)

    plot_timestamps = np.linspace(begin_timestamp, end_timestamp, 500)
    estimated_ps = estimated.position_curve.evaluate(plot_timestamps)
    vfusion_ps = vfusion_pos_curve.evaluate(plot_timestamps) - vfusion_pos_curve.evaluate(begin_timestamp)

    # Plot the estimated trajectory
    plt.clf()
    plt.plot(estimated_ps[:, 0], estimated_ps[:, 1], 'r-')
    plt.plot(vfusion_ps[:, 0], vfusion_ps[:, 1], 'b-')
    plt.axis('equal')
    plt.savefig('out/trajectory.pdf')

    # Plot the estimated trajectory at its own scale
    plt.clf()
    plt.plot(estimated_ps[:, 0], estimated_ps[:, 1], 'r-')
    plt.plot(estimated.position_curve.controls[:, 0], estimated.position_curve.controls[:, 1], 'b-', alpha=.2)
    plt.axis('equal')
    plt.savefig('out/lone_trajectory.pdf')

    # Plot the estimated vars
    plt.clf()
    plt.barh(np.arange(estimated.size), estimated.flatten(), height=.75, color='r')
    plt.savefig('out/vars.pdf')

    # Synthesize accel readings and compare to measured values
    timestamps = np.linspace(begin_timestamp, end_timestamp, 100)
    predicted_accel = []
    for t in timestamps:
        predicted_accel.append(predict_accel(estimated.position_curve,
                                             vfusion_orientation_curve,
                                             estimated.accel_bias,
                                             estimated.gravity,
                                             t))

    # Synthesize accel readings from gravity and accel bias only
    predicted_stationary_accel = []
    for t in timestamps:
        r = cayley.cayley(vfusion_orientation_curve.evaluate(t))
        predicted_stationary_accel.append(np.dot(r, estimated.gravity) + estimated.accel_bias)

    predicted_accel = np.array(predicted_accel)
    predicted_stationary_accel = np.array(predicted_stationary_accel)

    plt.clf()
    plt.plot(timestamps, predicted_accel, '-', label='predicted')
    plt.plot(accel_timestamps, accel_readings, '-', label='observed')
    plt.legend()
    plt.savefig('out/accel.pdf')

    plt.clf()
    plt.plot(timestamps, predicted_stationary_accel, '-', label='predicted')
    plt.plot(accel_timestamps, accel_readings, '-', label='observed')
    plt.legend()
    plt.savefig('out/accel_stationary.pdf')

    # Plot features
    plot_features(features, frame_timestamps, frame_orientations, estimated,
                  imu_to_camera, camera_matrix, 'out/estimated_features.pdf')


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
    #run_in_simulation()
    run_simulation_series()
    #run_with_dataset()
    #run_fit_spline()
    #run_fit_spline_multidim()
