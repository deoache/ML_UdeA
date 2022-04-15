import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend import plotting
from sklearn.tree import DecisionTreeClassifier, plot_tree


def decision_tree_viz(max_depth=1):
    data = sns.load_dataset("iris")
    X = data[["petal_length", "petal_width"]]
    y = data["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})

    dt_clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth).fit(
        X.values, y.values
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    sns.scatterplot(data=data, x="petal_length", y="petal_width", hue="species", ax=ax1)
    plotting.plot_decision_regions(X.values, y.values, dt_clf, legend=0, ax=ax1)
    plot_tree(
        dt_clf, feature_names=["petal_length", "petal_width"], filled=True, ax=ax2
    )

