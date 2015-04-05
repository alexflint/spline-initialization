import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns


def plot_accuracy_histograms():
    results = np.loadtxt('results/accuracy_comparison.txt').reshape((-1, 2, 4))

    pos_errs_socp, pos_errs_linear = results[:, :, 0].T
    vel_errs_socp, vel_errs_linear = results[:, :, 1].T
    bias_errs_socp, bias_errs_linear = results[:, :, 2].T
    g_errs_socp, g_errs_linear = results[:, :, 3].T

    c1, c2 = sns.color_palette("Set1", 2)

    xticks = range(-4, 1)
    xtick_labels = ['$10^{%d}$' % x for x in xticks]

    plt.clf()
    sns.kdeplot(np.log10(pos_errs_socp), shade=True, color=c1, label='SOCP')
    sns.kdeplot(np.log10(pos_errs_linear), shade=True, color=c2, label='Linear')
    plt.xlabel('$\log_{10}$ Device position error (m)')
    plt.xticks(xticks, xtick_labels)
    plt.ylabel('Frequency')
    plt.savefig('figures/position_error_histogram.pdf')

    plt.clf()
    sns.kdeplot(np.log10(vel_errs_socp), shade=True, color=c1, label='SOCP')
    sns.kdeplot(np.log10(vel_errs_linear), shade=True, color=c2, label='Linear')
    plt.xlabel('$\log_{10}$ Device velocity error (m/s)')
    plt.xticks(xticks, xtick_labels)
    plt.ylabel('Frequency')
    plt.savefig('figures/velocity_error_histogram.pdf')

    plt.clf()
    sns.kdeplot(np.log10(bias_errs_socp), shade=True, color=c1, label='SOCP')
    sns.kdeplot(np.log10(bias_errs_linear), shade=True, color=c2, label='Linear')
    plt.xlabel('$\log_{10}$ Accel bias error')
    plt.xticks(xticks, xtick_labels)
    plt.ylabel('Frequency')
    plt.savefig('figures/bias_error_histogram.pdf')

    plt.clf()
    sns.kdeplot(np.log10(np.rad2deg(g_errs_socp)), shade=True, color=c1, label='SOCP')
    sns.kdeplot(np.log10(np.rad2deg(g_errs_linear)), shade=True, color=c2, label='Linear')
    plt.xlabel('$\log_{10}$ Gravity error (degrees)')
    plt.xticks(xticks, xtick_labels)
    plt.ylabel('Frequency')
    plt.savefig('figures/gravity_error_histogram.pdf')


def plot_accuracy_vs_feature_noise():
    results = np.loadtxt('results/accuracy_vs_feature_noise.txt')
    feature_noise, socp_err = results.T

    c1, c2 = sns.color_palette("Set1", 2)

    plt.clf()
    #plt.errorbar(feature_noise, socp_mean, socp_std, color=c1, label='SOCP')
    #plt.errorbar(feature_noise, linear_mean, linear_std, color=c2, label='Linear')
    plt.plot(feature_noise, socp_err)
    plt.xlabel('Feature noise (image pixels)')
    plt.ylabel('Position error (m)')
    plt.ylim(ymin=0)
    plt.savefig('figures/accuracy_vs_feature_noise.pdf')


if __name__ == '__main__':
    #plot_accuracy_histograms()
    plot_accuracy_vs_feature_noise()
