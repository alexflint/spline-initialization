import numpy as np
import matplotlib.pyplot as plt

from geometry import arctans

def plot_segments(segments, *args, **kwargs):
    xs = []
    ys = []
    for segment in segments:
        xs += [ p[0] for p in segment ] + [None]
        ys += [ p[1] for p in segment ] + [None]
    return plt.plot(xs, ys, *args, **kwargs)


def plot_tracks(xs, *args, **kwargs):
    xs = np.asarray(xs)
    if np.shape(xs)[-1] == 3:
        #xs = pr(xs)
        xs = arctans(xs)
    xs = np.transpose(xs, (1, 0, 2))
    if 'limit' in kwargs:
        xs = xs[:kwargs.pop('limit')]
    plot_segments(xs, *args, **kwargs)

