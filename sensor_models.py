import numpy as np

import cayley
import geometry


def predict_accel(pos_curve, orient_curve, accel_bias, gravity, t):
    orientation = cayley.cayley(orient_curve.evaluate(t))
    return predict_accel_with_orientation(pos_curve, orientation, accel_bias, gravity, t)


def predict_accel_with_orientation(pos_curve, orientation, accel_bias, gravity, t):
    global_accel = pos_curve.evaluate_d2(t)
    return np.dot(orientation, global_accel + gravity) + accel_bias


def predict_feature(pos_curve, orient_curve, landmark, t, imu_to_camera, camera_matrix):
    p = pos_curve.evaluate(t)
    r = cayley.cayley(orient_curve.evaluate(t))
    return predict_feature_with_pose(r, p, landmark, imu_to_camera, camera_matrix)


def predict_feature_with_pose(r, p, x, imu_to_camera, camera_matrix, allow_behind=True):
    assert np.shape(r) == (3, 3), 'shape was '+str(np.shape(r))
    assert np.shape(p) == (3,), 'shape was '+str(np.shape(p))
    assert np.shape(x) == (3,), 'shape was '+str(np.shape(x))
    assert np.shape(imu_to_camera) == (3, 3), 'shape was '+str(np.shape(imu_to_camera))
    assert np.shape(camera_matrix) == (3, 3), 'shape was '+str(np.shape(camera_matrix))
    y = np.dot(camera_matrix, np.dot(imu_to_camera, np.dot(r, x - p)))
    if not allow_behind and y[2] <= 0:
        return None
    return geometry.pr(y)
