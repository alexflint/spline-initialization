import os

import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

import calibration


def enumerate_frame_files(path, skip=0):
    for entry in os.listdir(str(path)):
        if entry.endswith('.pgm'):
            if skip > 0:
                skip -= 1
            else:
                yield float(entry[:-4]), path/entry


def main():
    lens = calibration.LensModel(calibration.IPHONE_5S_CALIBRATION, scaling=.5)
    print lens.image_size
    print lens.image_to_calibrated([0, 0])
    print lens.image_to_calibrated([320, 180])
    return

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
        correspondences.append([(keypoints[0][m.queryIdx].pt, keypoints[i][m.trainIdx].pt) for m in matches])

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
    main()
