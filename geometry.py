import numpy as np
import scipy.optimize

from lie import SO3

def skew(m):
    m = np.asarray(m)
    return np.array([[0.,    -m[2],  m[1]],
                     [m[2],   0.,   -m[0]],
                     [-m[1],  m[0],    0.]])


def normalized(x):
    x = np.asarray(x)
    return x / np.linalg.norm(x)


def essential_residual(M):
    """Compute an error vector that is zero when M is an essential matrix."""
    r1 = np.linalg.det(M)
    MMT = np.dot(M, M.T)
    r2 = 2. * np.dot(MMT, M) - np.trace(MMT)*M
    return np.hstack((r1, r2.flatten()))


def essential_matrix(R, t):
    """Compute an error vector that is zero when M is an essential matrix."""
    return np.dot(R, skew(t))


def pr(x):
    x = np.asarray(x)
    return x[..., :-1] / x[..., -1:]


def unpr(x):
    if np.ndim(x) == 1:
        return np.hstack((x, 1))
    else:
        return np.hstack((x, np.ones((np.shape(x)[0], 1))))


def arctans(ps):
    ps = np.asarray(ps)
    return np.arctan2(ps[..., :-1], ps[..., -1:])


def pose_from_essential_matrix(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    U, s, Vt = np.linalg.svd(E)
    V = Vt.T
    R = np.dot(U, np.dot(W.T, V.T))
    skew_t = np.dot(V, np.dot(W, np.dot(np.diag(s), V.T)))
    t = np.array((skew_t[2, 1], skew_t[0, 2], skew_t[1, 0]))
    return R, t


def epipolar_error_from_pose(R, t, xs0, xs1):
    return epipolar_error(essential_matrix(R, t), xs0, xs1)


def epipolar_error(E, xs0, xs1):
    return sum(r*r for r in epipolar_residuals(E, xs0, xs1))


def epipolar_residuals(E, xs0, xs1):
    xs0 = unpr(xs0)
    xs1 = unpr(xs1)
    return [np.abs(np.dot(x1, np.dot(E, x0)))
            for x0, x1 in zip(xs0, xs1)]


def solve_essential_matrix_via_fmat(xs0, xs1, inlier_threshold):
    # Use opencv to solve for fundamental matrix
    F, inlier_mask = cv2.findFundamentalMat(xs0,
                                            xs1,
                                            method=cv2.FM_RANSAC,
                                            param1=inlier_threshold)

    assert F.shape == (3, 3)

    # Decompose and replace singular values
    u, _, v = np.linalg.svd(F)
    E = np.dot(u, np.dot(np.diag((1., 1., 0.)), v.T))
    E /= np.sum(E)

    return E, np.bool_(np.squeeze(inlier_mask))


def refine_epipolar_pose(R, t, xs0, xs1):
    ax = np.array([i == np.argmin(t) for i in range(3)], int)
    u = normalized(np.cross(ax, t))
    v = normalized(np.cross(u, t))

    def perturb_normalized(R, t, delta):
        assert len(delta) == 5
        return np.dot(SO3.exp(delta[:3]), R), normalized(t + delta[3]*u + delta[4]*v)

    def cost(delta):
        RR, tt = perturb_normalized(R, t, delta)
        return epipolar_error_from_pose(RR, tt, xs0, xs1)

    delta = scipy.optimize.fmin(cost, np.zeros(5), maxiter=500)
    RR, tt = perturb_normalized(R, t, delta)
    return RR, tt


def estimate_epipolar_pose(xs0, xs1, inlier_threshold, refine=True):
    # Solve for essential matrix using 8-point RANSAC
    E, inlier_mask = solve_essential_matrix_via_fmat(xs0, xs1, inlier_threshold)
    R, t = pose_from_essential_matrix(E)

    R = np.eye(3)  # temp hack

    # Polish pose using gradient descent
    if refine:
        RR, tt = refine_epipolar_pose(R, t, xs0[inlier_mask], xs1[inlier_mask])
    else:
        RR, tt = R, t

    # Report
    #print 'Num Inliers: %d (of %d)' % (np.sum(inlier_mask), len(xs0))
    #print 'R:\n', R
    #print 'RR:\n', RR
    #print 'Error after RANSAC:', epipolar_error_from_pose(R, t, xs0[inlier_mask], xs1[inlier_mask])
    #print 'Error after polishing:', epipolar_error_from_pose(RR, tt, xs0[inlier_mask], xs1[inlier_mask])

    return RR, tt, inlier_mask
