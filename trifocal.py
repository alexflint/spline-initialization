import numpy as np

from lie import SO3
import geometry


def normalized(x):
    x = np.asarray(x)
    return x / np.linalg.norm(x)


def main():
    num_landmarks = 5
    num_frames = 3

    landmarks = np.random.randn(num_landmarks, 3)
    positions = np.random.randn(num_frames, 3)
    orientations = map(SO3.exp, np.random.randn(num_frames, 3))

    positions[0,:] = 0
    orientations[0] = np.eye(3)

    poses = [np.hstack((r, -np.dot(r,p)[:,np.newaxis])) for r, p in zip(orientations, positions)]

    features = [[normalized(np.dot(r, x-p)) for r, p in zip(orientations, positions)]
                for x in landmarks]

    a = poses[1].T
    b = poses[2].T
    slices = [np.outer(a[i], b[3]) - np.outer(a[3], b[i]) for i in range(3)]

    x0, xa, xb = features[0]

    middle = sum(x0i*slice for x0i, slice in zip(x0, slices))
    residual = np.dot(geometry.skew(xa), np.dot(middle, geometry.skew(xb)))
    print 'Trifocal residual:', np.linalg.norm(residual)


if __name__ == '__main__':
    main()
