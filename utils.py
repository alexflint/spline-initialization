import collections

import numpy as np
from lie import skew, SO3


def normalized(x):
    x = np.asarray(x)
    return x / np.sqrt(np.sum(np.square(x), axis=-1))[..., None]


def pr(x):
    x = np.asarray(x)
    return x[..., :-1] / x[..., -1:]


def unpr(x):
    x = np.asarray(x)
    col_shape = x.shape[:-1] + (1,)
    return np.concatenate((x, np.ones(col_shape)), axis=-1)


def spy(x, tol=1e-4):
    x = np.atleast_2d(x)
    return '\n'.join(map(lambda row: '['+''.join('x' if abs(val)>tol else ' ' for val in row)+']', x))


def unreduce(x, mask, fill=0.):
    x = np.asarray(x)
    out = np.repeat(fill, len(mask))
    out[mask] = x
    return out


def unreduce_info(info, mask):
    out = np.zeros((len(mask), len(mask)))
    out[np.ix_(mask, mask)] = info
    return out


def cis(theta):
    """This works for both scalar and vector theta."""
    return np.array((np.cos(theta), np.sin(theta)))


def dots(*m):
    """Multiple an arbitrary number of matrices with np.dot."""
    return reduce(np.dot, m)


def sumsq(x, axis=None):
    """Compute the sum of squared elements."""
    return np.sum(np.square(x), axis=axis)


def unit(i, n):
    return (np.arange(n) == i).astype(float)


def orthonormalize(r):
    u, s, v = np.linalg.svd(r)
    return np.dot(u, v)


def minmedmax(xs):
    if len(xs) == 0:
        print 'warning [utils.minmedmax]: empty list passed'
        return 0., 0., 0.
    else:
        return np.min(xs), np.median(xs), np.max(xs)


def essential_matrix(R1, p1, R2, p2):
    Rrel = np.dot(R2, R1.T)
    prel = np.dot(R1, p2-p1)
    return essential_matrix_from_relative_pose(Rrel, prel)


def essential_matrix_from_relative_pose(Rrel, prel):
    return np.dot(Rrel, skew(prel))


def add_white_noise(x, sigma):
    return x + np.random.randn(*x.shape) * sigma


def add_orientation_noise(x, sigma):
    x = np.atleast_3d(x)
    return np.array([np.dot(xi, SO3.exp(np.random.randn(3)*sigma)) for xi in x])


def renumber_tracks(features, landmarks=None, min_track_length=None):
    # Drop tracks that are too short
    if min_track_length is not None:
        track_lengths = collections.defaultdict(int)
        for f in features:
            track_lengths[f.track_id] += 1
        features = filter(lambda f: track_lengths[f.track_id] >= min_track_length, features)

    # Apply consistent renumbering
    track_ids = sorted(set(f.track_id for f in features))
    track_index_by_id = {track_id: index for index, track_id in enumerate(track_ids)}
    for f in features:
        f.track_id = track_index_by_id[f.track_id]

    # Return the final features
    if landmarks is not None:
        assert len(track_ids) == 0 or len(landmarks) > max(track_ids)
        landmarks = np.array([landmarks[i] for i in track_ids])
        return features, landmarks
    else:
        return features
