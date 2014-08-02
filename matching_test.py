import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    img1 = cv2.imread('city1.png', cv2.CV_LOAD_IMAGE_COLOR)
    img2 = cv2.imread('city2.png', cv2.CV_LOAD_IMAGE_COLOR)

    #img2 = img1

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #features = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=.01, minDistance=4).squeeze()
    #features = features.astype(int)
    #img1[features[:,1], features[:,0]] = [0,0,255]

    #sift = cv2.SIFT()
    orb = cv2.ORB()
    kps1, des1 = orb.detectAndCompute(gray1, None)
    kps2, des2 = orb.detectAndCompute(gray2, None)

    #canvas = cv2.drawKeypoints(gray, kp)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    plt.clf()
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

    # Draw first 10 matches.
    #img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)
    #cv2.imshow('Matches', img3)
    #cv2.imshow('dst', img1)

    #if cv2.waitKey(0) & 0xff == 27:
        #cv2.destroyAllWindows()


    exit()





    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # Find corners
    ys, xs = np.nonzero(dst)
    corners = np.array((xs, ys)).T

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    refined_corners = corners.copy().astype(np.float32)
    cv2.cornerSubPix(np.uint8(gray), refined_corners, (5,5), (-1,-1), criteria)

    print 'Num initial corners:', len(corners)
    print 'Num refined corners:', len(refined_corners)

    dist = np.sqrt(np.sum(np.square(corners - refined_corners), axis=1))
    print 'Min distance: ', np.min(dist)
    print 'Max distance: ', np.max(dist)

    # Now draw them
    corners = np.int0(corners)
    refined_corners = np.int0(refined_corners)
    img1[corners[:,1], corners[:,0]] = [0,0,255]
    img1[refined_corners[:,1], refined_corners[:,0]] = [0,255,0]

    cv2.imshow('dst',img1)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

