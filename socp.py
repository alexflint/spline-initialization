import time
import numpy as np
import cvxopt as cx
import mosek

import lie
import geometry


timings = {}


class SocpConstraint(object):
    def __init__(self, a=None, b=None, c=None, d=None):
        if a is None:
            a = np.zeros(len(b), len(c))
        if b is None:
            b = np.zeros(len(a))
        if c is None:
            c = np.zeros(len(a[0]))
        if d is None:
            d = 0.
        assert np.ndim(a) == 2
        assert np.ndim(b) == 1
        assert np.ndim(c) == 1
        assert np.shape(a) == (len(b), len(c))
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.c = np.asarray(c)
        self.d = float(d)

    def conditionalize(self, mask, values):
        """Given a constraint over x1...xn, return a new constraint over a subset of the variables given fixed values
        for the remaining variables."""
        assert len(values) == sum(mask)
        mask = np.asarray(mask)
        values = np.asarray(values)
        a = self.a[:, ~mask]
        b = self.b + np.dot(self.a[:, mask], values)
        c = self.c[~mask]
        d = self.d + float(np.dot(self.c[mask], values))
        return SocpConstraint(a, b, c, d)

    def conditionalize_at_zero(self, mask):
        """Given a constraint over x1...xn, return a new constraint over a subset of the variables given fixed values
        for the remaining variables."""
        mask = np.asarray(mask)
        return SocpConstraint(self.a[:, ~mask], self.b, self.c[~mask], self.d)

    def lhs(self, x):
        return np.linalg.norm(np.dot(self.a, x) + self.b)

    def rhs(self, x):
        return np.dot(self.c, x) + self.d

    def is_satisfied(self, x):
        return self.lhs(x) <= self.rhs(x)


class SocpProblem(object):
    def __init__(self, objective, constraints=None):
        self.objective = np.asarray(objective)
        self.constraints = constraints or []

    def add_constraint(self, *args, **kwargs):
        self.constraints.append(SocpConstraint(*args, **kwargs))

    def conditionalize(self, mask, values=None):
        mask = np.asarray(mask)
        if values is None:
            return SocpProblem(self.objective[~mask], [x.conditionalize_at_zero(mask) for x in self.constraints])
        else:
            return SocpProblem(self.objective[~mask], [x.conditionalize(mask, values) for x in self.constraints])

    def conditionalize_indices(self, var_indices, values=None):
        if values is not None:
            assert len(var_indices) == len(values)
        mask = np.zeros(len(self.objective), bool)
        mask[np.array(var_indices)] = True
        return self.conditionalize(mask, values)

    def evaluate(self, x, verbose=False):
        print 'Objective:', np.dot(self.objective, x)
        lhs = np.array([constraint.lhs(x) for constraint in self.constraints])
        rhs = np.array([constraint.rhs(x) for constraint in self.constraints])
        num_violated = np.sum(lhs > rhs)

        if verbose or num_violated > 0:
            for i, (lhs, rhs) in enumerate(zip(lhs, rhs)):
                label = 'satisfied' if (lhs <= rhs) else 'not satisfied'
                print '  Constraint %d: %s (lhs=%.8f, rhs=%.8f)' % (i, label, lhs, rhs)

        if num_violated == 0:
            print '  All constraints satisfied'
        else:
            print '  Not satisfied (%d constraints violated)' % num_violated


def solve(problem, sparse=False, **kwargs):
    """Minimize w*x subject to ||Ax + b|| <= c*x + d."""
    gs = []
    hs = []
    for constraint in problem.constraints:
        a = constraint.a
        b = constraint.b
        c = constraint.c
        d = constraint.d
        g = np.vstack((-c, -a))
        hs.append(cx.matrix(np.hstack((d, b))))
        if sparse:
            gs.append(cx.sparse(cx.matrix(g)))
        else:
            gs.append(cx.matrix(g))
    begin = time.clock()
    cx.solvers.options.update(kwargs)
    cx.solvers.options['MOSEK'] = {mosek.iparam.log: 100, mosek.iparam.intpnt_max_iterations: 50000}
    solution = cx.solvers.socp(cx.matrix(problem.objective), Gq=gs, hq=hs, solver='mosek')
    duration = time.clock() - begin
    # duration = solution['duration']
    timings['last_solve'] = duration
    print 'SOCP duration: %.3f' % duration
    print 'Total duration (including python wrappers): %.3f' % duration
    print 'Solver exited with status "%s"' % solution['status']
    return solution


def run_2d_circle_problem():
    constraints = [
        SocpConstraint(np.eye(2), np.zeros(2), np.zeros(2), 3.),
        SocpConstraint(np.eye(2), [2, 0], np.zeros(2), 3.)
    ]
    problem = SocpProblem([0., -1.], constraints)
    sol = solve(problem)
    print sol['x']


def run_sfm():
    num_points = 4
    num_frames = 2
    num_vars = num_points * 3 + num_frames * 3

    r0 = np.eye(3)
    p0 = np.zeros(3)

    r1 = lie.SO3.exp([.1, .2, .3])
    p1 = np.array([2., 3., 0.])

    rs = [r0, r1]
    ps = [p0, p1]
    xs = np.random.randn(num_points, 3)

    vars = np.hstack(list(ps) + list(xs))

    gamma = 1e-6

    problem = SocpProblem(np.ones(num_vars), [])
    for i, x in enumerate(xs):
        for j, (r, p) in enumerate(zip(rs, ps)):
            z = geometry.pr(np.dot(r, x-p))

            position_offset = j*3
            point_offset = num_frames*3 + i*3

            a = np.zeros((2, num_vars))
            a[:, position_offset:position_offset+3] = np.outer(z, r[2]) - r[:2]
            a[:, point_offset:point_offset+3] = r[:2] - np.outer(z, r[2])

            sign = 1. if np.dot(r[2], x-p) >= 0 else -1.

            c = np.zeros(num_vars)
            c[position_offset:position_offset+3] = -sign * gamma * r[2]
            c[point_offset:point_offset+3] = sign * gamma * r[2]

            b = np.zeros(2)
            d = 0.

            ax = np.dot(a, vars)
            cx = np.dot(c, vars)
            print 'Point %d, camera %d:' % (i, j)
            print '  ax=', ax
            print '  cx=', cx

            problem.constraints.append(SocpConstraint(a, b, c, d))

    problem.evaluate(vars)
    #return

    structure_problem = problem.conditionalize(np.arange(num_vars) < 6, np.hstack(ps))

    sol = solve(structure_problem)
    print sol['x']
    if sol['x'] is None:
        print 'Solution not found'
    else:
        estimated_xs = np.array(sol['x']).reshape((-1, 3))
        print 'True:'
        print xs
        print 'Estimated:'
        print estimated_xs
        print 'Errors:'
        print xs - estimated_xs


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    #run_2d_circle_problem()
    run_sfm()
