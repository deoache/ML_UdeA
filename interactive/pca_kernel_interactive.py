from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import KernelPCA

import plotly.express as px


def pca_kernel_viz(
    kernel="linear", coef0=1, n_components=2, standarized=True, gamma=None, degree=2
):

    data = load_breast_cancer()
    X = data.data
    y = data.target

    pca = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
    )

    X_pca = (
        make_pipeline(StandardScaler(), pca).fit_transform(X)
        if standarized
        else pca.fit_transform(X)
    )

    labels = (
        {"x": "PC1", "y": "PC2"}
        if n_components == 2
        else {"x": "PC1", "y": "PC2", "z": "PC3"}
    )

    if n_components == 2:
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y, labels=labels)
    elif n_components == 3:
        fig = px.scatter_3d(
            x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], color=y, labels=labels
        )
    fig.show()
