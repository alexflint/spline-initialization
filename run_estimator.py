import bisect
import numpy as np

import matplotlib
matplotlib.use('Agg', warn=False)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns


import structures
import spline_socp
import spline
import utils


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
        z = sensor_models.predict_feature_with_pose(r, p, x, imu_to_camera, camera_matrix)
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
            all_features.append(
                structures.FeatureObservation(int(frame_id), int(track_id), np.array([float(x), float(y)])))

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
    vfusion_estimate = structures.PositionEstimate(vfusion_pos_curve, gravity, vfusion_accel_bias, vfusion_landmarks)

    vfusion_reproj_errors = compute_reprojection_errors(features, frame_timestamps, frame_orientations,
                                                        vfusion_estimate, imu_to_camera, camera_matrix)

    features = [f for f, err in zip(features, vfusion_reproj_errors) if np.linalg.norm(err) < 5.]
    features, vfusion_estimate.landmarks = utils.renumber_tracks(features, vfusion_estimate.landmarks, min_track_length=2)

    # Plot the reprojected landmarks
    plot_features(features, frame_timestamps, frame_orientations, vfusion_estimate,
                  imu_to_camera, camera_matrix, 'out/vfusion_features.pdf')

    # Create the problem
    print 'Creating problem for %d frames, %d tracks, %d features, and %d accel readings...' % \
          (len(frame_timestamps), len(track_ids), len(features), len(accel_readings))

    spline_tpl = spline.SplineTemplate.linspaced(num_knots, dims=3, begin=begin_timestamp, end=end_timestamp)
    estimator = 'mixed'
    if estimator == 'infnorm':
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
        predicted_accel.append(sensor_models.predict_accel(estimated.position_curve,
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
    #run_with_dataset()
    #run_fit_spline()
    #run_fit_spline_multidim()
