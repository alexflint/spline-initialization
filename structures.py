import numpy as np

import lie


class FeatureObservation(object):
    def __init__(self, frame_id, track_id, position):
        self.frame_id = frame_id
        self.track_id = track_id
        self.position = position


class Measurements(object):
    def __init__(self,
                 accel_timestamps,
                 accel_orientations,
                 accel_readings,
                 frame_timestamps,
                 frame_orientations,
                 features):
        self.accel_timestamps = accel_timestamps
        self.accel_orientations = accel_orientations
        self.accel_readings = accel_readings
        self.frame_timestamps = frame_timestamps
        self.frame_orientations = frame_orientations
        self.features = features


class Calibration(object):
    def __init__(self,
                 imu_to_camera,
                 camera_matrix,
                 gravity_magnitude):
        self.imu_to_camera = imu_to_camera
        self.camera_matrix = camera_matrix
        self.gravity_magnitude = gravity_magnitude

    @classmethod
    def random(cls, image_width=320., image_height=240., gravity_magnitude=9.8):
        imu_to_camera = lie.SO3.exp(np.random.randn(3))
        camera_matrix = np.array([[image_width, 0., image_width/2.],
                                  [0., image_height, image_height/2.],
                                  [0., 0., 1.]])
        return Calibration(imu_to_camera, camera_matrix, gravity_magnitude)


class PositionEstimate(object):
    def __init__(self, position_curve, gravity, accel_bias, landmarks):
        self.position_curve = position_curve
        self.gravity = gravity
        self.accel_bias = accel_bias
        self.landmarks = landmarks

    @property
    def size(self):
        return len(self.flatten())

    def flatten(self):
        return np.hstack((self.position_curve.controls.flatten(),
                          self.gravity,
                          self.accel_bias,
                          self.landmarks.flatten()))
