import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from lie import SO3
from geometry import pr, arctans
from utils import normalized, add_white_noise, add_orientation_noise
from bezier import zero_offset_bezier
from cayley import cayley
from plotting import plot_tracks

from estimate_orientation import estimate_orientation, predict_gyro, predict_orientation
from estimate_position import estimate_position, predict_accel, predict_feature


def run_position_estimation():
    #
    # Parameters
    #

    bezier_degree = 4
    num_frames = 8
    num_landmarks = 120
    num_accel_readings = 50
    num_gyro_readings = 60

    gyro_timestamp_noise = 0
    gyro_reading_noise = 1e-3

    accel_timestamp_noise = 0
    accel_reading_noise = 1e-3

    frame_timestamp_noise = 0
    frame_orientation_noise = 1e-3
    feature_noise = 1e-4

    print 'Num landmarks:', num_landmarks
    print 'Num frames:', num_frames
    print 'Num accel readings:', num_accel_readings
    print 'Num gyro readings:', num_gyro_readings
    print 'Bezier curve degree:', bezier_degree

    #
    # Construct ground truth
    #

    true_frame_timestamps = np.linspace(0, 1, num_frames)
    true_accel_timestamps = np.linspace(0, 1, num_accel_readings)

    true_gyro_bias = np.random.rand(3)
    true_accel_bias = np.random.randn(3)
    true_gravity_magnitude = 9.8
    true_gravity = normalized(np.random.rand(3)) * true_gravity_magnitude

    true_rot_controls = np.random.randn(bezier_degree-1, 3)
    true_pos_controls = np.random.randn(bezier_degree-1, 3)

    true_landmarks = np.random.randn(num_landmarks, 3) * 5
    true_landmarks[:, 2] += 20

    true_frame_orientations = np.array([cayley(zero_offset_bezier(true_rot_controls, t)) for t in true_frame_timestamps])
    true_frame_positions = np.array([zero_offset_bezier(true_pos_controls, t) for t in true_frame_timestamps])

    true_accel_readings = np.array([predict_accel(true_pos_controls, true_rot_controls, true_accel_bias, true_gravity, t)
                                    for t in true_accel_timestamps])

    true_features = np.array([[predict_feature(true_pos_controls, true_rot_controls, x, t) for x in true_landmarks]
                              for t in true_frame_timestamps])

    true_gyro_timestamps = np.linspace(0, 1, num_gyro_readings)
    true_gyro_readings = np.array([predict_gyro(true_rot_controls, true_gyro_bias, t)
                                   for t in true_gyro_timestamps])

    #
    # Add sensor noise
    #

    observed_gyro_timestamps = add_white_noise(true_gyro_timestamps, gyro_timestamp_noise)
    observed_gyro_readings = add_white_noise(true_gyro_readings, gyro_reading_noise)

    observed_accel_timestamps = add_white_noise(true_accel_timestamps, accel_timestamp_noise)
    observed_accel_readings = add_white_noise(true_accel_readings, accel_reading_noise)

    observed_frame_timestamps = add_white_noise(true_frame_timestamps, frame_timestamp_noise)
    observed_frame_orientations = add_orientation_noise(true_frame_orientations, frame_orientation_noise)
    observed_frame_orientations[0] = true_frame_orientations[0]  # do not add noise to first frame

    observed_features = add_white_noise(true_features, feature_noise)

    #
    # Plot features
    #
    plt.clf()
    plot_tracks(true_features, 'x-g', limit=10, alpha=.4)
    plot_tracks(observed_features, 'o-r', limit=10, alpha=.4)
    plt.show()
    return

    #
    #  Solve for orientation and gyro bias
    #

    print 'Estimating orientation...'
    estimated_gyro_bias, estimated_rot_controls = estimate_orientation(
        bezier_degree,
        observed_gyro_timestamps,
        observed_gyro_readings,
        observed_frame_timestamps,
        observed_frame_orientations)

    estimated_accel_orientations = np.array([predict_orientation(estimated_rot_controls, t)
                                             for t in observed_accel_timestamps])

    #
    # Solve for position, accel bias, and gravity
    #

    print 'Estimating position...'
    estimated_pos_controls, estimated_accel_bias, estimated_gravity = estimate_position(
        bezier_degree,
        observed_accel_timestamps,
        estimated_accel_orientations,
        observed_accel_readings,
        observed_frame_timestamps,
        observed_frame_orientations,
        observed_features)

    estimated_positions = np.array([zero_offset_bezier(estimated_pos_controls, t) for t in true_frame_timestamps])

    estimated_pfeatures = np.array([[pr(predict_feature(estimated_pos_controls, true_rot_controls, x, t))
                                     for x in true_landmarks]
                                    for t in true_frame_timestamps])
    true_pfeatures = pr(true_features)
    observed_pfeatures = pr(observed_features)

    #
    # Report
    #

    print 'Gyro bias error:', np.linalg.norm(estimated_gyro_bias - true_gyro_bias)
    print '  True gyro bias:', true_gyro_bias
    print '  Estimated gyro bias:', estimated_gyro_bias

    print 'Accel bias error:', np.linalg.norm(estimated_accel_bias - true_accel_bias)
    print '  True accel bias:', true_accel_bias
    print '  Estimated accel bias:', estimated_accel_bias

    print 'Gravity error:', np.linalg.norm(estimated_gravity - true_gravity)
    print '  True gravity:', true_gravity
    print '  Estimated gravity:', estimated_gravity
    print '  Estimated gravity magnitude:', np.linalg.norm(estimated_gravity)
    for i in range(num_frames):
        print 'Frame %d position error: %f' % (i, np.linalg.norm(estimated_positions[i] - true_frame_positions[i]))

    #
    # Plot orientation results
    #

    plot_timestamps = np.linspace(0, 1, 50)
    estimated_gyro_readings = np.array([predict_gyro(estimated_rot_controls, true_gyro_bias, t)
                                        for t in plot_timestamps])

    true_orientations = np.array([SO3.log(predict_orientation(true_rot_controls, t))
                                  for t in plot_timestamps])
    estimated_orientations = np.array([SO3.log(predict_orientation(estimated_rot_controls, t))
                                       for t in plot_timestamps])
    observed_orientations = np.array(map(SO3.log, observed_frame_orientations))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title('Gyro readings')
    plt.plot(plot_timestamps, estimated_gyro_readings, ':', label='estimated', alpha=1)
    plt.plot(true_gyro_timestamps, true_gyro_readings, '-', label='true', alpha=.3)
    plt.plot(true_gyro_timestamps, observed_gyro_readings, 'x', label='observed')
    plt.xlim(-.1, 1.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Orientation')
    plt.plot(plot_timestamps, estimated_orientations, ':', label='estimated', alpha=1)
    plt.plot(plot_timestamps, true_orientations, '-', label='true', alpha=.3)
    plt.plot(true_frame_timestamps, observed_orientations, 'x', label='observed')
    plt.xlim(-.1, 1.5)
    plt.legend()

    #
    # Plot position results
    #

    plot_timestamps = np.linspace(0, 1, 100)

    true_positions = np.array([zero_offset_bezier(true_pos_controls, t) for t in plot_timestamps])
    estimated_positions = np.array([zero_offset_bezier(estimated_pos_controls, t) for t in plot_timestamps])

    true_accels = np.array([predict_accel(true_pos_controls, true_rot_controls, true_accel_bias, true_gravity, t)
                            for t in plot_timestamps])
    estimated_accels = np.array([predict_accel(estimated_pos_controls, true_rot_controls, estimated_accel_bias, estimated_gravity, t)
                                 for t in plot_timestamps])

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], '-b')
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], '-r')
    #ax.plot(true_landmarks[:,0], true_landmarks[:,1], true_landmarks[:,2], '.k', alpha=.2)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(plot_timestamps, estimated_accels, ':', label='estimated', alpha=1)
    ax.plot(plot_timestamps, true_accels, '-', label='true', alpha=.3)
    ax.plot(observed_accel_timestamps, observed_accel_readings, 'x', label='observed')
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
