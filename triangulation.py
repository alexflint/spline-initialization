import numpy as np

import utils


def householder(x):
    """Compute a 2x3 matrix where the rows are orthogonal to x and orthogonal to each other."""
    assert len(x) == 3, 'x=%s' % x
    assert np.linalg.norm(x) > 1e-8
    a = (np.arange(3) == np.argmin(np.abs(x))).astype(float)
    u = utils.normalized(np.cross(x, a))
    v = utils.normalized(np.cross(x, u))
    return np.array([u, v])


def calibrated(z, k):
    """Compute the calibrated position for an image feature z."""
    assert k.shape == (3, 3)
    assert len(z) == 2
    return utils.normalized(np.linalg.solve(k, utils.unpr(z)))


def triangulate_midpoint(features, frame_orientations, frame_positions, imu_to_camera, camera_matrix):
    """Triangulate a landmark from two or more views using the midpoint method."""
    assert len(features) > 0
    jtj, jtr = np.zeros((3, 3)), np.zeros(3)
    for f in features:
        r = frame_orientations[f.frame_id]
        p = frame_positions[f.frame_id]
        h = householder(calibrated(f.position, camera_matrix))
        a = utils.dots(h, imu_to_camera, r)
        b = -utils.dots(h, imu_to_camera, r, p)
        jtj += np.dot(a.T, a)
        jtr += np.dot(a.T, b)
    return -np.linalg.solve(jtj, jtr)
