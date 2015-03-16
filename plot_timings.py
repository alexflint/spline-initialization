import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns


def main():
    num_landmarks, duration = np.loadtxt('results/timings_vs_num_landmarks.txt').T
    plt.clf()
    plt.plot(num_landmarks, duration)
    plt.xlabel('Number of landmarks')
    plt.ylabel('Solve duration (s)')
    plt.ylim(ymin=0)
    plt.savefig('figures/timings_vs_num_landmarks.pdf')

    num_knots, duration = np.loadtxt('results/timings_vs_num_knots.txt').T
    plt.clf()
    plt.plot(num_knots, duration)
    plt.xlabel('Number of spline knots')
    plt.ylabel('Solve duration (s)')
    plt.ylim(0, .1)
    plt.savefig('figures/timings_vs_num_knots.pdf')


if __name__ == '__main__':
    main()
