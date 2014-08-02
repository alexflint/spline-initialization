import numpy as np


def perspective_distorted_to_calibrated(distorted, radial_params, tangential_params, num_iters=20):
    k1, k2, k3 = radial_params
    p1, p2 = tangential_params

    undistorted = np.asarray(distorted).copy()
    for i in range(num_iters):
        r_2 = np.dot(undistorted, undistorted)
        k_radial = 1. + k1 * r_2 + k2 * r_2*r_2 + k3 * r_2 * r_2 * r_2
        delta_x_0 = 2*p1*undistorted[0]*undistorted[1] + p2*(r_2 + 2*undistorted[0]*undistorted[0])
        delta_x_1 = p1 * (r_2 + 2*undistorted[1]*undistorted[1])+2*p2*undistorted[0]*undistorted[1]
        undistorted[0] = (distorted[0]-delta_x_0) / k_radial
        undistorted[1] = (distorted[1]-delta_x_1) / k_radial

    return undistorted


def perspective_image_to_calibrated(image, camera_matrix_inv, radial_params, tangential_params, num_iters=20):
    image = np.asarray(image)
    if len(image) == 2:
        image = np.array([image[0], image[1], 1.])
    distorted = np.dot(camera_matrix_inv, image)
    return perspective_distorted_to_calibrated(distorted, radial_params, tangential_params, num_iters)


class LensModel(object):
    def __init__(self, calibration, scaling=1., **kwargs):
        self.calibration = calibration
        self.image_size = (np.array(calibration['image_size']) * scaling).astype(int)
        self.camera_matrix = np.array(calibration['camera_matrix']) * scaling
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

    def image_to_calibrated(self, image):
        return perspective_image_to_calibrated(image,
                                               self.camera_matrix_inv,
                                               self.calibration['radial_params'],
                                               self.calibration['tangential_params'])


IPHONE_5S_CALIBRATION = dict(
    image_size=(1280,720),
    camera_matrix=np.array([[1097.0328, 0, 635.60778],
                            [0., 1099.82306, 371.75292],
                            [0., 0., 1.]]),
    radial_params=[0.06788, -0.07547, 0.],
    tangential_params=[0., 0.],
    camera_to_imu_rotation=np.array([[-0.00046152, -0.99999178, -0.00402814],
                                     [-0.99997218, 0.00043151, 0.00744678],
                                     [-0.00744498, 0.00403146, -0.99996416]])
)



