import numpy as np

import copy
import structures
import simulation
import socp
import spline_socp
import utils


simulator_options = dict(
    duration=5.,
    num_frames=8,
    num_landmarks=50,
    num_imu_readings=100,
    degree=3,
    num_controls=8,
    accel_timestamp_noise=0,
    accel_reading_noise=1e-3,
    accel_orientation_noise=0,
    frame_timestamp_noise=0,
    frame_orientation_noise=0,
    feature_noise=1.
)


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


def evaluate(calibration, measurements, spline_template, estimator, tol, true_trajectory):
    estimated_trajectory = spline_socp.estimate_trajectory(
        calibration,
        measurements,
        spline_template,
        estimator=estimator,
        feature_tolerance=tol)
    pos_err = mean_position_error(true_trajectory, estimated_trajectory, measurements.frame_timestamps)
    vel_err = mean_velocity_error(true_trajectory, estimated_trajectory, measurements.frame_timestamps)
    bias_err = accel_bias_error(true_trajectory, estimated_trajectory)
    g_err = gravity_direction_error(true_trajectory, estimated_trajectory)
    return pos_err, vel_err, bias_err, g_err


def simulate_and_evaluate(num_trials, calibration, estimators=None, **options):
    if estimators is None:
        estimators = ['mixed']
    trials = []
    while len(trials) < num_trials:
        try:
            true_trajectory, measurements, spline_template = simulation.simulate_trajectory(
                calibration, **options)
            row = []
            for estimator in estimators:
                row.extend(evaluate(
                    calibration,
                    measurements,
                    spline_template,
                    estimator,
                    simulator_options['feature_noise']*3,
                    true_trajectory))
            trials.append(row)
        except spline_socp.FeasibilityError:
            print 'Simulator failed to generate trajectory. Retrying...'
        except spline_socp.InsufficientObservationsError:
            print 'Simulator failed to generate trajectory. Retrying...'
    return np.asarray(trials)


def run_accuracy_comparison():
    np.random.seed(0)
    calibration = structures.Calibration.random()
    trials = simulate_and_evaluate(1000, calibration, ['mixed', 'householder'], **simulator_options)
    np.savetxt('results/accuracy_comparison.txt', trials)


def run_accuracy_vs_feature_noise():
    np.random.seed(1)
    calibration = structures.Calibration.random()

    options = simulator_options.copy()
    options['feature_noise'] = 0.
    options['accel_reading_noise'] = 1e-2

    true_trajectory, measurements, spline_template = simulation.simulate_trajectory(calibration, **options)

    results = []
    for feature_noise in np.linspace(0, 10, 25):
        print 'Trying feature noise = %f' % feature_noise
        noisy_measurements = copy.deepcopy(measurements)
        for f in noisy_measurements.features:
            f.position += np.random.randn(2) * feature_noise

        try:
            pos_err, vel_err, bias_err, g_err = evaluate(
                calibration,
                noisy_measurements,
                spline_template,
                'mixed',
                feature_noise*3+1e-3,
                true_trajectory)

            results.append((feature_noise, pos_err))
        except spline_socp.FeasibilityError:
            pass

    np.savetxt('results/accuracy_vs_feature_noise.txt', results)


def run_timings_vs_num_landmarks():
    np.random.seed(0)

    options = simulator_options.copy()
    options['num_landmarks'] = 1000

    calibration = structures.Calibration.random()

    trials = []
    true_trajectory, measurements, spline_template, true_frame_timestamps = simulation.simulate_trajectory(
        calibration, **options)
    all_features = measurements.features
    for n in np.linspace(10, 400, 25):
        measurements.features = filter(lambda f: f.track_id < n, all_features)
        try:
            spline_socp.estimate_trajectory(
                calibration,
                measurements,
                spline_template,
                estimator='mixed',
                feature_tolerance=options['feature_noise']*3)
            trials.append((n, socp.timings['last_solve']))
        except spline_socp.InsufficientObservationsError:
            print 'Simulator failed to generate trajectory. Retrying...'

    np.savetxt('results/timings_vs_num_landmarks.txt', trials)


def run_timings_vs_num_knots():
    np.random.seed(0)

    calibration = structures.Calibration.random()

    trials = []
    true_trajectory, measurements, spline_template, true_frame_timestamps = simulation.simulate_trajectory(
        calibration, **simulator_options)
    for n in np.arange(2, 21):
        spline_template.knots = np.linspace(0, simulator_options['duration'], n)
        try:
            spline_socp.estimate_trajectory(
                calibration,
                measurements,
                spline_template,
                estimator='mixed',
                feature_tolerance=simulator_options['feature_noise']*3)
            trials.append((n, socp.timings['last_solve']))
        except spline_socp.InsufficientObservationsError:
            print 'Simulator failed to generate trajectory. Retrying...'

    np.savetxt('results/timings_vs_num_knots.txt', trials)


if __name__ == '__main__':
    run_accuracy_vs_feature_noise()
    #run_accuracy_comparison()
    #run_timings_vs_num_knots()
