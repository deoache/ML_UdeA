import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


def svm_soft_margin_viz(C):
  # data scatterplot
  penguins = sns.load_dataset("penguins")
  penguins = penguins.loc[penguins.species != "Gentoo"].dropna()

  X = penguins[["bill_length_mm", "flipper_length_mm"]]
  y = penguins["species"].replace({"Adelie":0, "Chinstrap":1})

  fig, ax = plt.subplots(figsize=(10, 7))
  sns.scatterplot(x=X["bill_length_mm"], y=X["flipper_length_mm"], hue=penguins["species"], ax=ax)

  # classifier training
  scaler = StandardScaler()
  clf = SVC(kernel="linear", C=C)
  svm_clf = make_pipeline(scaler, clf).fit(X, y)
  ax.set_title(f"C: {C}", size=15)

  # decision regions and margin
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  x1_grid = np.linspace(xlim[0], xlim[1])
  x2_grid = np.linspace(ylim[0], ylim[1])

  xx, yy = np.meshgrid(x1_grid, x2_grid)
  r1, r2 = np.c_[xx.flatten()], np.c_[yy.flatten()]
  grid = pd.DataFrame(np.hstack((r1, r2)), columns=X.columns)

  z_decision_function = svm_clf.decision_function(grid).reshape(xx.shape)
  z_predict = svm_clf.predict(grid).reshape(xx.shape)

  ax.contour(xx, yy, z_decision_function, colors="k", levels=[-1, 0, 1], linestyles=["--", "-", "--"], alpha=0.5)
  ax.contourf(xx, yy, z_predict, cmap='Paired', alpha=0.2)

  # support vectors
  support_vectors = scaler.inverse_transform(svm_clf["svc"].support_vectors_)
  ax.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', s=60, color="k")
