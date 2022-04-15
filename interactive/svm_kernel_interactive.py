import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_moons,
)

from interactive_utils import (
    get_grid,
    plot_decision_regions,
)


def kernel_trick_viz(dataset="blobs", kernel="linear", C=1, degree=2, coef0=1, gamma=1):
    # data scatterplot
    n_samples = 500
    noise = 0.08
    random_state = 42
    if "blobs" in dataset:
        centers = [(-5, -10), (9, 10), (-1, 0), (9, 0)]
        X, y = make_blobs(
            n_samples=n_samples, centers=centers, random_state=random_state
        )
        y = np.array([1 if label == 0 or label == 1 else 0 for label in y])
    elif "circles" in dataset:
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=0.1, random_state=random_state
        )
    elif "moons" in dataset:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax)

    # classifier training
    svm_clf = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0, degree=degree),
    ).fit(X, y)

    # decision regions
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy, grid = get_grid(xlim, ylim)
    plot_decision_regions(svm_clf, xx, yy, grid, ax)

