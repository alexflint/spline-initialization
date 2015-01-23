import numpy as np

import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt
import seaborn

import utils


def diagify(xs, dims):
    """Given a rank-1 vector xs, return a DIMS x n*DIMS array where each DIMS x DIMS subblock is a diagonal matrix
    with xs[i] repeated along the diagonal."""
    return np.hstack([np.eye(dims)*x for x in xs])


def num_bases(num_knots, degree):
    """Compute the number of basis functions for a spline with the given degree and number of knots."""
    return num_knots + degree - 1


def basis(ts, i, knots, degree):
    """Evaluate the i-th B-spline basis function at T."""
    n = len(knots)
    if degree == 0:
        if i == 0:
            return (ts <= knots[1]).astype(float)
        elif i+2 == n:
            return (ts > knots[-2]).astype(float)
        else:
            return np.logical_and(knots[i] <= ts, ts < knots[i+1]).astype(float)
    else:
        out = 0.
        if i > 0:
            coef = (ts - knots[max(i-degree, 0)]) / (knots[min(i, n-1)] - knots[max(i-degree, 0)])
            out += coef * basis(ts, i-1, knots, degree-1)
        if i + 1 < n + degree - 1:
            coef = (knots[min(i+1, n-1)] - ts) / (knots[min(i+1, n-1)] - knots[max(i-degree+1, 0)])
            out += coef * basis(ts, i, knots, degree-1)
        return out


def basis_d1(ts, i, knots, degree):
    """Evaluate the first derivative of the i-th B-spline basis function at T."""
    n = len(knots)
    if degree == 0:
        return 0 if np.isscalar(ts) else np.zeros(len(ts))
    else:
        out = 0.
        if i > 0:
            denom = 1. / (knots[min(i, n-1)] - knots[max(i-degree, 0)])
            coef = (ts - knots[max(i-degree, 0)]) * denom
            out += coef * basis_d1(ts, i-1, knots, degree-1) + denom * basis(ts, i-1, knots, degree-1)
        if i + 1 < n + degree - 1:
            denom = 1. / (knots[min(i+1, n-1)] - knots[max(i-degree+1, 0)])
            coef = (knots[min(i+1, n-1)] - ts) * denom
            out += coef * basis_d1(ts, i, knots, degree-1) - denom * basis(ts, i, knots, degree-1)
        return out


def basis_d2(ts, i, knots, degree):
    """Evaluate the first derivative of the i-th B-spline basis function at T."""
    n = len(knots)
    if degree == 0:
        return 0 if np.isscalar(ts) else np.zeros(len(ts))
    else:
        out = 0.
        if i > 0:
            denom = 1. / (knots[min(i, n-1)] - knots[max(i-degree, 0)])
            coef = (ts - knots[max(i-degree, 0)]) * denom
            out += coef*basis_d2(ts, i-1, knots, degree-1) + 2*denom*basis_d1(ts, i-1, knots, degree-1)
        if i + 1 < n + degree - 1:
            denom = 1. / (knots[min(i+1, n-1)] - knots[max(i-degree+1, 0)])
            coef = (knots[min(i+1, n-1)] - ts) * denom
            out += coef*basis_d2(ts, i, knots, degree-1) - 2*denom*basis_d1(ts, i, knots, degree-1)
        return out


def coefficients(ts, knots, degree):
    """Compute the coefficients of all bases for a spline evaluated at T."""
    return np.array([basis(ts, i, knots, degree)
                     for i in range(num_bases(len(knots), degree))]).T


def coefficients_d1(ts, knots, degree):
    """Compute the coefficients of all bases for a spline evaluated at T."""
    return np.array([basis_d1(ts, i, knots, degree)
                     for i in range(num_bases(len(knots), degree))]).T


def coefficients_d2(ts, knots, degree):
    """Compute the coefficients of all bases for a spline evaluated at T."""
    return np.array([basis_d2(ts, i, knots, degree)
                     for i in range(num_bases(len(knots), degree))]).T


def multidim_coefficients(t, knots, degree, dims):
    """For splines through multidimensional space, compute a matrix A such that dot(A, controls.flatten()) is the
    spline output at T."""
    return diagify(coefficients(t, knots, degree), dims)


def multidim_coefficients_d1(t, knots, degree, dims):
    """For splines through multidimensional space, compute a matrix A such that dot(A, controls.flatten()) is the
    spline output at T."""
    return diagify(coefficients_d1(t, knots, degree), dims)


def multidim_coefficients_d2(t, knots, degree, dims):
    """For splines through multidimensional space, compute a matrix A such that dot(A, controls.flatten()) is the
    spline output at T."""
    return diagify(coefficients_d2(t, knots, degree), dims)


def evaluate(ts, knots, degree, controls):
    """Evaluate a spline at T."""
    return np.dot(coefficients(ts, knots, degree), controls)


def evaluate_d1(ts, knots, degree, controls):
    """Evaluate a spline at T."""
    return np.dot(coefficients_d1(ts, knots, degree), controls)


def evaluate_d2(ts, knots, degree, controls):
    """Evaluate a spline at T."""
    return np.dot(coefficients_d2(ts, knots, degree), controls)


class SplineTemplate(object):
    """Represents the knots and degree for a B-spline."""
    def __init__(self, knots, degree, dims=1):
        self.knots = knots
        self.degree = degree
        self.dims = dims

    @property
    def num_bases(self):
        return num_bases(len(self.knots), self.degree)

    @property
    def control_shape(self):
        return (self.num_bases,) if self.dims == 1 else (self.num_bases, self.dims)

    @property
    def control_size(self):
        return self.num_bases * self.dims

    def build_random(self, scale=1., offset=0., first_control=None):
        controls = np.random.randn(*self.control_shape)*scale + offset
        if first_control is not None:
            controls[0] = first_control
        return Spline(self, controls)

    def coefficients(self, ts):
        return coefficients(ts, self.knots, self.degree)

    def coefficients_d1(self, ts):
        return coefficients_d1(ts, self.knots, self.degree)

    def coefficients_d2(self, ts):
        return coefficients_d2(ts, self.knots, self.degree)

    def multidim_coefficients(self, ts):
        return multidim_coefficients(ts, self.knots, self.degree, self.dims)

    def multidim_coefficients_d1(self, ts):
        return multidim_coefficients_d1(ts, self.knots, self.degree, self.dims)

    def multidim_coefficients_d2(self, ts):
        return multidim_coefficients_d2(ts, self.knots, self.degree, self.dims)

    def evaluate(self, ts, controls):
        return evaluate(ts, self.knots, self.degree, controls)

    def evaluate_d1(self, ts, controls):
        return evaluate_d1(ts, self.knots, self.degree, controls)

    def evaluate_d2(self, ts, controls):
        return evaluate_d2(ts, self.knots, self.degree, controls)

    @classmethod
    def linspaced(cls, num_knots, dims, duration=1., degree=3, begin=0., end=None):
        if end is None:
            end = begin + duration
        return SplineTemplate(np.linspace(begin, end, num_knots), degree, dims)


class Spline(object):
    """Represents a B-spline curve."""
    def __init__(self, template, controls):
        controls = np.asarray(controls)
        assert isinstance(template, SplineTemplate)
        if template.dims == 1:
            assert len(controls) == template.num_bases
        else:
            assert controls.shape == template.control_shape, '%s vs %s' % (controls.shape, template.control_shape)
        self.template = template
        self.controls = controls

    def evaluate(self, ts):
        """Evaluate the spline at T."""
        return self.template.evaluate(ts, self.controls)

    def evaluate_d1(self, ts):
        """Evaluate the first derivative of the spline at T."""
        return self.template.evaluate_d1(ts, self.controls)

    def evaluate_d2(self, ts):
        """Evaluate the second derivative of the spline at T."""
        return self.template.evaluate_d2(ts, self.controls)

    @classmethod
    def canonical(cls, controls, duration=1., degree=3, begin=0., end=None):
        """Construct a spline with uniformly spaced knots from the specified control points."""
        num_knots = len(controls) - degree + 1
        dims = np.shape(controls)[1] if np.ndim(controls) > 1 else 1
        tpl = SplineTemplate.linspaced(num_knots, dims, degree=degree, duration=duration, begin=begin, end=end)
        return Spline(tpl, controls)


def fit(ts, ys, degree=3, num_knots=None, knot_frequency=5.):
    ts = np.asarray(ts)
    ys = np.asarray(ys)
    t0 = ts[0]
    duration = ts[-1] - t0
    dims = 1 if np.ndim(ys) == 1 else ys.shape[1]
    if num_knots is None:
        num_knots = int(np.ceil(duration * knot_frequency)) + 1

    # Create the linear system
    tpl = SplineTemplate(np.linspace(t0, t0+duration, num_knots), degree, dims)
    a = np.vstack([tpl.multidim_coefficients(t) for t in ts])
    b = ys.flatten()

    # Solve the system
    controls, _, _, _ = np.linalg.lstsq(a, b)

    # Construct the spline
    if dims > 1:
        controls = controls.reshape((-1, dims))
    return Spline(tpl, controls)


def main():
    np.random.seed(2)

    degree = 3
    num_knots = 8
    knots = sorted([0, 10] + list(np.random.rand(num_knots-2) * 10.))

    plt.clf()
    for i in range(num_knots + degree - 1):
        ts = np.linspace(-.1, 10.1, 200)
        ys = basis(ts, i, knots, degree)
        plt.plot(ts, ys)
    plt.vlines(knots, -1, 2, linestyles='dotted', alpha=.4)
    plt.savefig('out/bases.pdf')


if __name__ == '__main__':
    main()
