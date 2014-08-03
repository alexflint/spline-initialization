import os

import numpy as np
import scipy.optimize
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

import calibration
import geometry
from lie import SO3


def enumerate_frame_files(path, skip=0):
    for entry in os.listdir(str(path)):
        if entry.endswith('.pgm'):
            if skip > 0:
                skip -= 1
            else:
                yield float(entry[:-4]), path/entry


def plot_correspondences(img1, img2, kps1, kps2, matches, style):
    plt.set_cmap('gray')
    plt.imshow(img1, extent=(0, .9, 1, 0))
    plt.imshow(img2, extent=(1, 1.9, 1, 0))
    plt.xlim(0, 1.9)
    plt.ylim(1, 0)

    for match in matches:
        kp1 = kps1[match.queryIdx]
        kp2 = kps2[match.trainIdx]
        x1, y1 = kp1.pt
        x2, y2 = kp2.pt

        x1 = .9 * x1 / img1.shape[1]
        y1 = y1 / img1.shape[0]

        x2 = 1. + .9 * x2 / img2.shape[1]
        y2 = y2 / img2.shape[0]

        plt.plot([x1,x2], [y1,y2], style)


def main():
    np.random.seed(123)

    # Load calibration
    lens = calibration.LensModel(calibration.IPHONE_5S_CALIBRATION, scaling=.5)
    epipolar_threshold = lens.image_distance_to_calibrated(2.)
    print 'Epipolar threshold:', epipolar_threshold

    # Load images
    skip_frames = 45
    data_dir = Path('/Users/alexflint/Data/Initialization/Painting')
    all_frame_timestamps, all_frame_paths = zip(*enumerate_frame_files(data_dir / 'imageDump',
                                                                       skip=skip_frames))

    # Select frames to track
    #frame_indices = np.linspace(0, len(all_frame_timestamps)-1, num_frames).round().astype(int)
    frame_indices = range(0, 60, 5)
    num_frames = len(frame_indices)

    frame_timestamps = [all_frame_timestamps[i] for i in frame_indices]
    frame_orientations = [np.eye(3)]

    # Create descriptor
    orb = cv2.ORB()

    # Compute keypoints and descriptors
    timestamps = []
    color_images = []
    gray_images = []
    keypoints = []
    descriptors = []
    for i in frame_indices:
        image = cv2.imread(str(all_frame_paths[i]), cv2.CV_LOAD_IMAGE_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        color_images.append(image)
        gray_images.append(gray)
        timestamps.append(all_frame_timestamps[i])

        kps, descr = orb.detectAndCompute(gray, None)
        keypoints.append(kps)
        descriptors.append(descr)

    # Create exhaustive matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Initialize tracks
    raw_tracks = [[(0, kp.pt[0], kp.pt[1])] for kp in keypoints[0]]
    inlier_tracks = [[(0, kp.pt[0], kp.pt[1])] for kp in keypoints[0]]

    # Match features between frame 1 and frame i
    for i in range(1, num_frames):
        print 'Processing frame %d of %d' % (i, num_frames-1)
        matches = bf.match(descriptors[0], descriptors[i])

        # Sort matches in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = [keypoints[0][m.queryIdx].pt for m in matches]
        pts2 = [keypoints[i][m.trainIdx].pt for m in matches]

        pts1 = np.asarray(map(lens.image_to_calibrated, pts1))
        pts2 = np.asarray(map(lens.image_to_calibrated, pts2))

        #xmin, ymin = np.min(pts1, axis=0)
        #xmax, ymax = np.max(pts1, axis=0)
        #print 'X: %f ... %f' % (xmin, xmax)
        #print 'Y: %f ... %f' % (ymin, ymax)

        R, t, inlier_mask = geometry.estimate_epipolar_pose(pts1, pts2, epipolar_threshold, refine=True)
        print R
        frame_orientations.append(R)

        inlier_matches = []
        outlier_matches = []
        for match, is_inlier in zip(matches, inlier_mask):
            kp = keypoints[i][match.trainIdx]
            raw_tracks[match.queryIdx].append((i, kp.pt[0], kp.pt[1]))
            if is_inlier:
                inlier_matches.append(match)
                inlier_tracks[match.queryIdx].append((i, kp.pt[0], kp.pt[1]))
            else:
                outlier_matches.append(match)

        plt.clf()
        plt.hold('on')
        plot_correspondences(gray_images[0], gray_images[i], keypoints[0], keypoints[i], outlier_matches, 'r-')
        plot_correspondences(gray_images[0], gray_images[i], keypoints[0], keypoints[i], inlier_matches, 'b-')
        plt.show()


    def compute_correspondence_matrix(tracks, num_frames):
        mat = np.zeros((num_frames, num_frames), int)
        for track in tracks:
            for fi, xi, yi in track:
                for fj, xj, yj in track:
                    mat[fi, fj] += 1
        return mat

    raw_cmatrix = compute_correspondence_matrix(raw_tracks, num_frames)
    inlier_cmatrix = compute_correspondence_matrix(inlier_tracks, num_frames)

    print 'Raw correspondence counts:'
    print raw_cmatrix
    np.savetxt('out/raw_cmatrix.txt', raw_cmatrix, fmt='%3d')

    print 'Inlier correspondence counts:'
    print inlier_cmatrix
    np.savetxt('out/inlier_cmatrix.txt', inlier_cmatrix, fmt='%3d')

    np.savetxt('out/frame_orientations.txt',
               np.hstack((np.asarray(frame_timestamps)[:,None],
                          np.asarray(frame_orientations).reshape((-1, 9)))))

    accel_data = np.loadtxt(str(data_dir / 'accelerometer.txt'))
    begin_time = frame_timestamps[0]
    end_time = frame_timestamps[-1]
    accel_mask = np.logical_and(begin_time < accel_data[:,0], accel_data[:,0] < end_time)
    accel = accel_data[accel_mask]
    np.savetxt('out/accelerometer.txt', accel)

    with open('out/features.txt', 'w') as fd:
        for i, track in enumerate(inlier_tracks):
            if len(track) >= 2:
                for feature in track:
                    fd.write('%d %d %f %f\n' % (i, feature[0], feature[1], feature[2]))


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    main()
