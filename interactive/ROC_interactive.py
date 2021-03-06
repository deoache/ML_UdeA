from interactive_utils import plot_probability_distributions, plot_roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def roc_curve_viz(pos_dist_mean=0.25, neg_dist_mean=0.75):
    # probability distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))
    plot_probability_distributions(pos_dist_mean, neg_dist_mean, ax=ax1)

    # threshold and decision regions
    threshold = 0.5
    ax1.vlines(threshold, 0, 5)
    ax1.axvspan(0, threshold, alpha=0.1, color="blue")
    ax1.axvspan(threshold, 1, alpha=0.1, color="darkorange")
    ax1.axis([0, 1, 0, 5])
    ax1.legend()

    # ROC curve
    plot_roc_curve(pos_dist_mean, neg_dist_mean, ax=ax2)
