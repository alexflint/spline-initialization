import numpy as np


def perspective_distorted_to_calibrated(distorted, radial_params, tangential_params, num_iters=20):
    assert len(distorted) == 2
    k1, k2, k3 = radial_params
    p1, p2 = tangential_params

    undistorted = np.asarray(distorted).copy()
    for i in range(num_iters):
        r2 = np.dot(undistorted, undistorted)
        k_radial = 1. + k1 * r2 + k2 * r2*r2 + k3 * r2 * r2 * r2
        delta_x_0 = 2.*p1*undistorted[0]*undistorted[1] + p2*(r2 + 2.*undistorted[0]*undistorted[0])
        delta_x_1 = p1*(r2 + 2.*undistorted[1]*undistorted[1]) + 2.*p2*undistorted[0]*undistorted[1]
        undistorted[0] = (distorted[0]-delta_x_0) / k_radial
        undistorted[1] = (distorted[1]-delta_x_1) / k_radial

    return undistorted


def perspective_image_to_calibrated(image, camera_matrix_inv, radial_params, tangential_params, num_iters=20):
    image = np.asarray(image)
    if len(image) == 2:
        image = np.array([image[0], image[1], 1.])
    distorted = np.dot(camera_matrix_inv, image)
    distorted = distorted[:2] / distorted[2]
    #return distorted
    return perspective_distorted_to_calibrated(distorted, radial_params, tangential_params, num_iters)


class LensModel(object):
    def __init__(self, calibration, scaling=1., **kwargs):
        self.calibration = calibration
        self.image_size = (np.array(calibration['image_size']) * scaling).astype(int)
        self.camera_matrix = np.array(calibration['camera_matrix']) * scaling
        print scaling
        print self.camera_matrix
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

    def image_to_calibrated(self, image):
        return perspective_image_to_calibrated(image,
                                               self.camera_matrix_inv,
                                               self.calibration['radial_params'],
                                               self.calibration['tangential_params'])

    def image_distance_to_calibrated(self, distance):
        """Compute a distance in the calibrated domain equivalent to the specified
        distance in the image domain. This is an approximation and only applies near
        the center of the image."""
        return np.mean(np.diag(self.camera_matrix_inv)[:2] * distance)


# Taken from:
# https://github.com/FlybyMedia/Nestor/blob/master/data/camera/iPhone5640h360wparameters_Oleg.xml
IPHONE_5S_CALIBRATION = dict(
    image_size=(1280,720),
    camera_matrix=np.array([[625.39885, 0, 321.32716],
                            [0., 624.71256, 175.33386],
                            [0., 0., 1.]]),
    radial_params=[0.08238, -0.03432, 0.],
    tangential_params=[0., 0.],
    camera_to_imu_rotation=np.array([[-0.99731099, -0.0675295, -0.02847041],
                                     [0.06712113, -0.9976311, 0.01506464],
                                     [-0.02942027, 0.01311316, 0.99948111]])
)



