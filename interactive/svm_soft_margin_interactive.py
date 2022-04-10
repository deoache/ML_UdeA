import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


def interactive_svm_viz(C):
    # data scatterplot
    penguins = sns.load_dataset("penguins")
    penguins = penguins.loc[penguins.species != "Gentoo"].dropna()

    X = penguins[["bill_length_mm", "flipper_length_mm"]]
    y = penguins["species"]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x=X["bill_length_mm"], y=X["flipper_length_mm"], hue=y, ax=ax)

    # classifier
    scaler = StandardScaler()
    clf = SVC(kernel="linear", C=C)
    svm_clf = make_pipeline(scaler, clf).fit(X, y)
    ax.set_title(f"C: {C}", size=15)
    
    # decision frontier and margin
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1])
    yy = np.linspace(ylim[0], ylim[1])

    YY, XX = np.meshgrid(yy, xx)
    xy = pd.DataFrame(np.vstack([XX.ravel(), YY.ravel()]).T, columns=X.columns)
    Z = svm_clf.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    
    # support vectors
    support_vectors = scaler.inverse_transform(svm_clf["svc"].support_vectors_)
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', s=60, color="k")