import numpy as np
import numpy.testing as tst

import unittest
import numdifftools

import spline



class SplineTest(unittest.TestCase):
    def test_derivative_degree2(self):
        np.random.seed(0)
        knots = np.linspace(0, 10, 8)
        t0 = 3.5
        f = lambda t: spline.basis(t, 2, knots, degree=2)
        j_numeric = numdifftools.Derivative(f)(t0)
        j_analytic = spline.basis_d1(t0, 2, knots, degree=2)
        self.assertAlmostEqual(j_numeric, j_analytic)

    def test_derivative_degree3(self):
        np.random.seed(0)
        knots = np.linspace(0, 10, 8)
        t0 = 2.
        f = lambda t: spline.basis(t, 2, knots, degree=3)
        j_numeric = numdifftools.Derivative(f)(t0)
        j_analytic = spline.basis_d1(t0, 2, knots, degree=3)
        self.assertAlmostEqual(j_numeric, j_analytic)

    def test_derivative_degree3_t0(self):
        np.random.seed(0)
        knots = np.linspace(0, 10, 8)
        t0 = .5
        f = lambda t: spline.basis(t, 0, knots, degree=3)
        j_numeric = numdifftools.Derivative(f)(t0)
        j_analytic = spline.basis_d1(t0, 0, knots, degree=3)
        self.assertAlmostEqual(j_numeric, j_analytic)

    def test_derivative2_degree2(self):
        np.random.seed(0)
        knots = np.linspace(0, 10, 8)
        t0 = 3.5
        f = lambda t: spline.basis(t, 2, knots, degree=2)
        h_numeric = numdifftools.Derivative(f, 2)(t0)
        h_analytic = spline.basis_d2(t0, 2, knots, degree=2)
        self.assertAlmostEqual(h_numeric, h_analytic)

    def test_derivative2_degree3(self):
        np.random.seed(0)
        knots = np.linspace(0, 10, 8)
        t0 = 2.
        f = lambda t: spline.basis(t, 2, knots, degree=3)
        h_numeric = numdifftools.Derivative(f, 2)(t0)
        h_analytic = spline.basis_d2(t0, 2, knots, degree=3)
        self.assertAlmostEqual(h_numeric, h_analytic)

    def test_derivative2_degree3_t0(self):
        np.random.seed(0)
        knots = np.linspace(0, 10, 8)
        t0 = .5
        f = lambda t: spline.basis(t, 0, knots, degree=3)
        h_numeric = numdifftools.Derivative(f, 2)(t0)
        h_analytic = spline.basis_d2(t0, 0, knots, degree=3)
        self.assertAlmostEqual(h_numeric, h_analytic)

    def test_evaluate(self):
        np.random.seed(0)
        curve = spline.Spline.canonical(np.random.randn(8))
        j_analytic = curve.evaluate_d1(.5)
        j_numeric = numdifftools.Derivative(curve.evaluate)(.5)
        h_analytic = curve.evaluate_d2(.5)
        h_numeric = numdifftools.Derivative(curve.evaluate, 2)(.5)
        self.assertAlmostEqual(j_numeric, j_analytic)
        self.assertAlmostEqual(h_numeric, h_analytic)

    def test_evaluate_multidim(self):
        np.random.seed(0)
        curve = spline.Spline.canonical(np.random.randn(8, 3))

        # First derivative
        j_analytic = curve.evaluate_d1(.5)
        j_numeric = np.squeeze(numdifftools.Jacobian(curve.evaluate)(.5))
        tst.assert_array_almost_equal(j_numeric, j_analytic)

        # Second derivative
        # We should be able to use numdifftools.Hessian below but I could not get it to work
        h_analytic = curve.evaluate_d2(.5)
        h_numeric = np.squeeze([numdifftools.Derivative(lambda t: curve.evaluate(t)[i], 2)(.5) for i in range(3)])
        tst.assert_array_almost_equal(h_numeric, h_analytic)
