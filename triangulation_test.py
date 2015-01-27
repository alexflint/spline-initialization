import unittest
import numpy as np
import numpy.testing

import utils
import lie
import triangulation
import spline_socp


class TriangulationTest(unittest.TestCase):
    def test_triangulate_midpoint(self):
        np.random.seed(0)
        for num_frames in [2, 3, 10]:
            for noise, decimals in [(0, 12), (1e-8, 8), (1e-3, 3), (1e-2, 1)]:
                ps = np.random.randn(num_frames, 3)
                rs = map(lie.SO3.exp, np.random.randn(num_frames, 3)*.1)

                imu_to_camera = lie.SO3.exp(np.random.randn(3)*.1)
                camera_matrix = np.array([[100, 0, 50],
                                          [0, 100, 50],
                                          [0, 0, 1]], dtype=float)

                x = np.random.randn(3) + [0, 0, 10]

                features = []
                for i, (r, p) in enumerate(zip(rs, ps)):
                    z = utils.pr(utils.dots(camera_matrix, imu_to_camera, r, x - p))
                    if noise > 0:
                        z += np.random.randn(2) * noise
                    features.append(spline_socp.FeatureObservation(i, 0, z))

                estimated = triangulation.triangulate_midpoint(features, rs, ps, imu_to_camera, camera_matrix)
                numpy.testing.assert_array_almost_equal(estimated, x, decimal=decimals)
