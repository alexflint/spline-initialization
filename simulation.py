import numpy as np

import spline
import utils
import cayley
import geometry
import sensor_models
import structures


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
    true_accel_readings = np.array([
        sensor_models.predict_accel(true_pos_curve, true_rot_curve, true_accel_bias, true_gravity, t)
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
                true_features.append(structures.FeatureObservation(frame_id, track_id, geometry.pr(y)))
            else:
                num_behind += 1

    if num_behind > 0:
        print '%d landmarks were behind the camera (and %d were in front)' % (num_behind, len(true_features))

    true_features, true_landmarks = utils.renumber_tracks(true_features, true_landmarks, min_track_length=2)
    true_trajectory = structures.PositionEstimate(true_pos_curve, true_gravity, true_accel_bias, true_landmarks)

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
        observed_features.append(structures.FeatureObservation(f.frame_id,
                                                               f.track_id,
                                                               utils.add_white_noise(f.position, feature_noise)))

    measurements = structures.Measurements(observed_accel_timestamps,
                                           observed_accel_orientations,
                                           observed_accel_readings,
                                           observed_frame_timestamps,
                                           observed_frame_orientations,
                                           observed_features)

    return true_trajectory, measurements, spline_template
