import os

import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

import calibration
from lie import SO3


def skew(m):
    m = np.asarray(m)
    return np.array([[0.,    -m[2],  m[1]],
                     [m[2],   0.,   -m[0]],
                     [-m[1],  m[0],    0.]])


def essential_residual(M):
    """Compute an error vector that is zero when M is an essential matrix."""
    r1 = np.linalg.det(M)
    MMT = np.dot(M, M.T)
    r2 = 2. * np.dot(MMT, M) - np.trace(MMT)*M
    return np.hstack((r1, r2.flatten()))


def essential_matrix(R, t):
    """Compute an error vector that is zero when M is an essential matrix."""
    return np.dot(R, skew(t))


def enumerate_frame_files(path, skip=0):
    for entry in os.listdir(str(path)):
        if entry.endswith('.pgm'):
            if skip > 0:
                skip -= 1
            else:
                yield float(entry[:-4]), path/entry


def pr(x):
    x = np.atleast_2d(x)
    return x[:,:-1] / x[:,-1:]


def unpr(x):
    if np.ndim(x) == 1:
        return np.hstack((x, 1))
    else:
        return np.hstack((x, np.ones(np.shape(x)[0])))


def pose_from_essential(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    U, s, Vt = np.linalg.svd(E)
    V = Vt.T
    R = np.dot(U, np.dot(W.T, V.T))
    skew_t = np.dot(V, np.dot(W, np.dot(np.diag(s), V.T)))
    t = np.array((skew_t[2, 1], skew_t[0, 2], skew_t[1, 0]))
    return R, t


def epipolar_error_from_pose(R, t, xs0, xs1):
    return epipolar_error(essential_matrix(R, t), xs0, xs1)


def epipolar_error(E, xs0, xs1):
    cost = 0.
    for x0, x1 in zip(xs0, xs1):
        cost += np.dot(unpr(x1), np.dot(E, unpr(x0)))
    return cost


def main():
    np.random.seed(123)

    lens = calibration.LensModel(calibration.IPHONE_5S_CALIBRATION, scaling=.5)

    #R = SO3.exp(np.random.rand(3))
    #t = np.random.rand(3)
    #E = np.dot(R, skew(t))
    #print essential_residual(E)
    #return

    data_dir = Path('/Users/alexflint/Data/Initialization/Apt')

    all_frame_timestamps, all_frame_paths = zip(*enumerate_frame_files(data_dir / 'imageDump', skip=5))

    num_frames = 5
    indices = np.linspace(0, len(all_frame_timestamps)-1, num_frames).round().astype(int)

    # create descriptor
    orb = cv2.ORB()

    timestamps = []
    color_images = []
    gray_images = []
    keypoints = []
    descriptors = []
    for i in indices:
        image = cv2.imread(str(all_frame_paths[i]), cv2.CV_LOAD_IMAGE_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        color_images.append(image)
        gray_images.append(gray)
        timestamps.append(all_frame_timestamps[i])

        kps, descr = orb.detectAndCompute(gray, None)
        keypoints.append(kps)
        descriptors.append(descr)

    # create exhaustive matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match features between frame 1 and frame i
    correspondences = []
    for i in range(1, num_frames):
        matches = bf.match(descriptors[0], descriptors[i])

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = [keypoints[0][m.queryIdx].pt for m in matches]
        pts2 = [keypoints[i][m.trainIdx].pt for m in matches]
        correspondences.append(zip(pts1, pts2))

        print lens.image_to_calibrated([640, 360])

        pts1 = np.asarray(map(lens.image_to_calibrated, pts1))
        pts2 = np.asarray(map(lens.image_to_calibrated, pts2))

        print pts1.shape
        print pts2.shape

        F, inliners = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC)

        F /= np.sum(F)

        xmin, ymin = np.min(pts1, axis=0)
        xmax, ymax = np.max(pts1, axis=0)
        print 'X: %f ... %f' % (xmin, xmax)
        print 'Y: %f ... %f' % (ymin, ymax)

        U, S, V = np.linalg.svd(F)
        print 'U:'
        print U
        print 'S:'
        print S
        print 'V:'
        print V

        E = np.dot(U, np.dot(np.diag((1., 1., 0.)), V.T))
        E /= np.sum(E)

        print 'Num Inliers:', np.sum(inliners)
        print 'Residual:', np.linalg.norm(essential_residual(F))
        print 'Residual:', np.linalg.norm(essential_residual(E))

        print 'Error:', epipolar_error(F, pts1, pts2)
        print 'Error:', epipolar_error(E, pts1, pts2)

        return


        img1 = gray_images[0]
        img2 = gray_images[i]
        kps1 = keypoints[0]
        kps2 = keypoints[i]

        plt.clf()
        plt.set_cmap('gray')
        plt.hold('on')
        plt.imshow(img1, extent=(0, .9, 1, 0))
        plt.imshow(img2, extent=(1, 1.9, 1, 0))
        plt.xlim(0, 1.9)
        plt.ylim(1, 0)

        for match in matches[:50]:
            kp1 = kps1[match.queryIdx]
            kp2 = kps2[match.trainIdx]
            x1, y1 = kp1.pt
            x2, y2 = kp2.pt

            x1 = .9 * x1 / img1.shape[1]
            y1 = y1 / img1.shape[0]

            x2 = 1. + .9 * x2 / img2.shape[1]
            y2 = y2 / img2.shape[0]

            print match.queryIdx, match.trainIdx
            print x1, y1, x2, y2

            plt.plot([x1,x2], [y1,y2], 'r-')


        plt.show()
        break


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    main()
