import bisect
import numpy as np

import utils
import cayley
import spline
import socp
import sensor_models
import lie
import structures


class InsufficientObservationsError(Exception):
    pass


class FeasibilityError(Exception):
    pass


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


class FirstOrderRotationCurve(object):
    def __init__(self, timestamps, orientations):
        self.timestamps = timestamps
        self.orientations = orientations

    def evaluate(self, t):
        return cayley.cayley_inv(interpolate_orientation(self.timestamps, self.orientations, t))


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
        prediction = sensor_models.predict_accel_with_orientation(
            trajectory.position_curve,
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
        z = sensor_models.predict_feature_with_pose(r, p, x, imu_to_camera, camera_matrix)
        errors.append(z - feature.position)
    return np.array(errors)


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
    if len(observed_features) < 5:
        raise InsufficientObservationsError()

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
        raise FeasibilityError('Solver returned status "%s"' % result['status'])

    estimated_vars = np.hstack((np.zeros(3), np.squeeze(result['x'])))

    spline_vars = spline_template.control_size
    pos_controls = estimated_vars[:spline_vars].reshape((-1, 3))
    gravity = estimated_vars[spline_vars:spline_vars+3]
    accel_bias = estimated_vars[spline_vars+3:spline_vars+6]
    landmarks = estimated_vars[spline_vars+6:].reshape((-1, 3))

    curve = spline.Spline(spline_template, pos_controls)
    return structures.PositionEstimate(curve, gravity, accel_bias, landmarks)


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
        raise FeasibilityError('Solver returned status "%s"' % result['status'])

    estimated_vars = np.hstack((np.zeros(3), np.squeeze(result['x'])))

    spline_vars = spline_template.control_size
    pos_controls = estimated_vars[:spline_vars].reshape((-1, 3))
    gravity = estimated_vars[spline_vars:spline_vars+3]
    accel_bias = estimated_vars[spline_vars+3:spline_vars+6]
    landmarks = estimated_vars[spline_vars+6:-1].reshape((-1, 3))

    curve = spline.Spline(spline_template, pos_controls)
    return structures.PositionEstimate(curve, gravity, accel_bias, landmarks)


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
    return structures.PositionEstimate(position_curve, gravity, accel_bias, landmarks)


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
    return structures.PositionEstimate(position_curve, gravity, accel_bias, landmarks)


def estimate_trajectory(calibration,
                        measurements,
                        spline_template,
                        estimator='mixed',
                        feature_tolerance=5.,
                        accel_tolerance=.1,
                        ground_truth=None):
    if estimator == 'socp':
        raise ValueError("'socp' has been renamed to 'infnorm' to avoid confusion with 'mixed'")
    elif estimator == 'infnorm':
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
        raise Exception('Invalid solver:'+str(estimator))
