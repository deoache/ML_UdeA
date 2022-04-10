import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy.integrate as integrate
import scipy.stats as stats

### roc curve utils
def get_xlims(mean, sigma=0.082):
  """compute normal distribution x limits"""
  xmin = mean - 3 * sigma
  xmax = mean + 3 * sigma
  return xmin, xmax 

def get_normal_dist(mean, sigma=0.082):
  """compute normal distribution coordinates"""
  xmin, xmax = get_xlims(mean, sigma)
  x = np.linspace(xmin, xmax, 100)
  y = stats.norm.pdf(x, mean, sigma)
  return x, y

def get_area(min, max, mean, sigma=0.082):
  """compute the area under a normal distribution"""
  f = lambda x, mean: stats.norm.pdf(x, mean, sigma)
  area = integrate.quad(f, min, max, args=(mean))[0]
  return area

def fill_region(mean, min, max, sigma=0.082, label=None, color=None, ax=None):
  """fill some region of a normal distribution between min and max"""
  x = np.linspace(min, max, 100)
  y = stats.norm.pdf(x, mean, sigma)
  ax.fill_between(x, y, facecolor=color, alpha=0.2, label=label)

def fill_false_regions(pos_dist_mean, neg_dist_mean, threshold=0.5, ax=None):
  """plot false and positive regions"""
  _, pos_dist_max = get_xlims(pos_dist_mean)
  neg_dist_min, _ = get_xlims(neg_dist_mean)

  pos_dist_x, pos_dist = get_normal_dist(pos_dist_mean) 
  neg_dist_x, neg_dist = get_normal_dist(neg_dist_mean)

  if neg_dist_min < threshold:
    fill_region(
        mean=neg_dist_mean, 
        min=neg_dist_min, 
        max=threshold,
        label="FP",
        color="r",
        ax=ax
        )
      
  if pos_dist_max > threshold:
    fill_region(
        mean=pos_dist_mean, 
        min=threshold, 
        max=pos_dist_max,
        label="FN",
        color="b",
        ax=ax
        )

def plot_probability_distributions(pos_dist_mean, neg_dist_mean, threshold=0.5, ax=None):
  """
  plot one-dimensional probability distributions for a binary classifier
  """
  pos_dist_x, pos_dist = get_normal_dist(pos_dist_mean) 
  neg_dist_x, neg_dist = get_normal_dist(neg_dist_mean)

  ax.plot(pos_dist_x, pos_dist, label="Positive class")
  ax.plot(neg_dist_x, neg_dist, label="Negative class")
  ax.set_title("Probability distributions")
  fill_false_regions(pos_dist_mean, neg_dist_mean, threshold, ax)

@np.vectorize
def get_classification_results(pos_dist_mean, neg_dist_mean, threshold):
  """compute true and false positive/negative values"""
  pos_dist_min, pos_dist_max = get_xlims(pos_dist_mean)
  neg_dist_min, neg_dist_max = get_xlims(neg_dist_mean)
    
  # area under a normal curve
  auc = get_area(0, 0.5, 0.25)
  
  # positive class scenarios
  if threshold <= pos_dist_min:
    TP = 0
    FN = auc
  elif threshold >= pos_dist_max:
    TP = auc
    FN = 0
  else:
    TP = get_area(pos_dist_min, threshold, pos_dist_mean)
    FN = get_area(threshold, pos_dist_max, pos_dist_mean)

  # negative class scenarios
  if threshold <= neg_dist_min:
    TN = auc
    FP = 0
  elif threshold >= neg_dist_max:
    TN = 0
    FP = auc
  else:
    TN = get_area(threshold, neg_dist_max, neg_dist_mean)
    FP = get_area(neg_dist_min, threshold, neg_dist_mean)

  return TP, TN, FP, FN

@np.vectorize
def get_positive_rates(pos_dist_mean, neg_dist_mean, threshold):
  """compute false and positive rates"""
  TP, TN, FP, FN = get_classification_results(pos_dist_mean, neg_dist_mean, threshold)
  fpr = FP / (FP + TN + 1e-5)
  tpr = TP / (TP + FN + 1e-5)
  return fpr, tpr

def plot_roc_curve(pos_dist_mean, neg_dist_mean, ax):
  # false and positive rates
  threshold = np.linspace(0, 1)
  fpr, tpr = get_positive_rates(pos_dist_mean, neg_dist_mean, threshold)
  
  # area under the curve
  auc = np.trapz(tpr, fpr)

  ax.set(
      title="ROC Curve", 
      xlabel="False positive rate", 
      ylabel="True positive rate"
      )
  ax.plot([0, 1], [0, 1], "k--")
  ax.plot(fpr, tpr, "r", label=f"AUC {auc:.2f}")
  ax.legend()

### svm soft margin utils
def get_grid(xlim: tuple, ylim: tuple, columns: list = None):
  """compute grid points"""
  x1_grid = np.linspace(xlim[0], xlim[1])
  x2_grid = np.linspace(ylim[0], ylim[1])

  xx, yy = np.meshgrid(x1_grid, x2_grid)
  r1, r2 = np.c_[xx.flatten()], np.c_[yy.flatten()]
  if columns is None:
    grid = np.hstack((r1, r2))
  else:
    grid = pd.DataFrame(np.hstack((r1, r2)), columns=columns)

  return xx, yy, grid

def plot_decision_regions(clf, x_grid, y_grid, grid, ax):
  z = clf.predict(grid).reshape(x_grid.shape)
  ax.contourf(x_grid, y_grid, z, cmap='Paired', alpha=0.2)

def plot_margins(clf, x_grid, y_grid, grid, ax):
  z = clf.decision_function(grid).reshape(x_grid.shape)
  ax.contour(x_grid, y_grid, z, colors="k", levels=[-1, 0, 1], linestyles=["--", "-", "--"], alpha=0.5)