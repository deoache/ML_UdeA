import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def precision_recall_viz(treshold=0):
  # one-dimensional data for a binary classification
  n = 10
  positive_class = pd.DataFrame(dict(x=np.linspace(-3, 10, n), y=np.ones(n, dtype=int)))
  negative_class = pd.DataFrame(dict(x=np.linspace(-10, 3, n), y=np.zeros(n, dtype=int)))

  plt.figure(figsize=(12, 7))
  sns.scatterplot(
      x=positive_class.x, 
      y=0.2 * np.ones(n), 
      label="Positive class"
      )
  sns.scatterplot(
      x=negative_class.x, 
      y=-0.2 * np.zeros(n), 
      label="Negative class"
      )

  # precision and recall
  tp = len(positive_class[positive_class.x >= treshold])
  tn = len(negative_class[negative_class.x < treshold])
  fp = len(negative_class[negative_class.x > treshold])
  fn = len(positive_class[positive_class.x <= treshold])

  precision = tp / (tp + fp)
  recall = tp / (tp + fn)

  plt.text(3, -1.5, f"TP: {tp} FP: {fp}", size=15)
  plt.text(-8, -1.5, f"TN: {tn} FN: {fn}", size=15)
  plt.title(f"Precision: {precision:.2f} | Recall: {recall:.2f}", size=17)
  
  # tresholds and decision regions
  plt.vlines(treshold, -2, 3, label="Treshold")
  plt.axvspan(treshold, 11, alpha=0.2)
  plt.axvspan(-11, treshold, alpha=0.2, color="darkorange")
  
  plt.axis([-11, 11, -2, 3])
  plt.legend()