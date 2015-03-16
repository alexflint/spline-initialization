import numpy as np

import structures
import simulation
import socp
import spline_socp
import utils


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


def run_accuracy_trials():
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

    num_trials = 10

    calibration = structures.Calibration.random()

    trials = []
    while len(trials) < num_trials:
        try:
            true_trajectory, measurements, spline_template, true_frame_timestamps = simulation.simulate_trajectory(
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
                estimated_trajectory = spline_socp.estimate_trajectory(calibration,
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
        except spline_socp.InsufficientObservationsError:
            print 'Simulator failed to generate trajectory. Retrying...'

    np.savetxt('results/trials.txt', trials)


def run_timings_vs_num_landmarks():
    np.random.seed(0)

    duration = 5.
    num_frames = 8
    num_landmarks = 1000
    num_imu_readings = 100
    degree = 3
    num_controls = 8
    accel_timestamp_noise = 0
    accel_reading_noise = 1e-3
    accel_orientation_noise = 0
    frame_timestamp_noise = 0
    frame_orientation_noise = 0
    feature_noise = 1.

    calibration = structures.Calibration.random()

    trials = []
    true_trajectory, measurements, spline_template, true_frame_timestamps = simulation.simulate_trajectory(
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
    all_features = measurements.features
    for n in np.linspace(10, 400, 25):
        measurements.features = filter(lambda f: f.track_id < n, all_features)
        try:
            spline_socp.estimate_trajectory(calibration,
                                            measurements,
                                            spline_template,
                                            estimator='mixed',
                                            feature_tolerance=feature_noise*3)
            trials.append((n, socp.timings['last_solve']))
        except spline_socp.InsufficientObservationsError:
            print 'Simulator failed to generate trajectory. Retrying...'

    np.savetxt('results/timings_vs_num_landmarks.txt', trials)


def run_timings_vs_num_knots():
    np.random.seed(0)

    duration = 5.
    num_frames = 8
    num_landmarks = 200
    num_imu_readings = 100
    degree = 3
    num_controls = 8
    accel_timestamp_noise = 0
    accel_reading_noise = 1e-3
    accel_orientation_noise = 0
    frame_timestamp_noise = 0
    frame_orientation_noise = 0
    feature_noise = 1.

    calibration = structures.Calibration.random()

    trials = []
    true_trajectory, measurements, spline_template, true_frame_timestamps = simulation.simulate_trajectory(
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
    for n in np.arange(2, 21):
        spline_template.knots = np.linspace(0, duration, n)
        try:
            spline_socp.estimate_trajectory(calibration,
                                            measurements,
                                            spline_template,
                                            estimator='mixed',
                                            feature_tolerance=feature_noise*3)
            trials.append((n, socp.timings['last_solve']))
        except spline_socp.InsufficientObservationsError:
            print 'Simulator failed to generate trajectory. Retrying...'

    np.savetxt('results/timings_vs_num_knots.txt', trials)


if __name__ == '__main__':
    run_timings_vs_num_knots()
