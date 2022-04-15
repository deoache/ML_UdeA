from interactive_utils import get_grid, plot_decision_regions, plot_margins
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def svm_soft_margin_viz(C):
    # data scatterplot
    penguins = sns.load_dataset("penguins")
    penguins = penguins.loc[penguins.species != "Gentoo"].dropna()

    X = penguins[["bill_length_mm", "flipper_length_mm"]]
    y = penguins["species"]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        x=X["bill_length_mm"], y=X["flipper_length_mm"], hue=penguins["species"], ax=ax
    )

    # classifier training
    scaler = StandardScaler()
    clf = SVC(kernel="linear", C=C)
    svm_clf = make_pipeline(scaler, clf).fit(X, y)
    ax.set_title(f"C: {C}", size=15)

    # decision regions and margin
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy, grid = get_grid(xlim, ylim, X.columns)
    plot_margins(svm_clf, xx, yy, grid, ax)
    plot_decision_regions(svm_clf, xx, yy, grid, ax)

    # support vectors
    support_vectors = scaler.inverse_transform(svm_clf["svc"].support_vectors_)
    ax.scatter(
        support_vectors[:, 0], support_vectors[:, 1], facecolors="none", s=60, color="k"
    )
