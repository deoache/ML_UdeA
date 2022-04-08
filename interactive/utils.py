import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy.integrate as integrate
import scipy.stats as stats
import ipywidgets as widgets


# generate one-dimensional probability distributions for a binary classifier
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

def fill_false_regions(pos_dist_mean, neg_dist_mean, treshold=0.5, ax=None):
  """plot false and positive regions"""
  _, pos_dist_max = get_xlims(pos_dist_mean)
  neg_dist_min, _ = get_xlims(neg_dist_mean)

  pos_dist_x, pos_dist = get_normal_dist(pos_dist_mean) 
  neg_dist_x, neg_dist = get_normal_dist(neg_dist_mean)

  if neg_dist_min < treshold:
    fill_region(
        mean=neg_dist_mean, 
        min=neg_dist_min, 
        max=treshold,
        label="FP",
        color="r",
        ax=ax
        )
      
  if pos_dist_max > treshold:
    fill_region(
        mean=pos_dist_mean, 
        min=treshold, 
        max=pos_dist_max,
        label="FN",
        color="b",
        ax=ax
        )

def plot_probability_distributions(pos_dist_mean, neg_dist_mean, treshold=0.5, ax=None):
  """
  plot one-dimensional probability distributions for a binary classifier
  """
  pos_dist_x, pos_dist = get_normal_dist(pos_dist_mean) 
  neg_dist_x, neg_dist = get_normal_dist(neg_dist_mean)

  ax.plot(pos_dist_x, pos_dist, label="Positive class")
  ax.plot(neg_dist_x, neg_dist, label="Negative class")
  ax.set_title("Probability distributions")
  fill_false_regions(pos_dist_mean, neg_dist_mean, treshold, ax)

@np.vectorize
def get_positive_rates(pos_dist_mean, neg_dist_mean, treshold):
  """compute false and positive rates"""
  pos_dist_min, pos_dist_max = get_xlims(pos_dist_mean)
  neg_dist_min, neg_dist_max = get_xlims(neg_dist_mean)
    
  # area under a normal curve
  auc = get_area(0, 0.5, 0.25)
  
  # positive class scenarios
  if treshold <= pos_dist_min:
    TP = 0
    FN = auc
  elif treshold >= pos_dist_max:
    TP = auc
    FN = 0
  else:
    TP = get_area(pos_dist_min, treshold, pos_dist_mean)
    FN = get_area(treshold, pos_dist_max, pos_dist_mean)

  # negative class scenarios
  if treshold <= neg_dist_min:
    TN = auc
    FP = 0
  elif treshold >= neg_dist_max:
    TN = 0
    FP = auc
  else:
    TN = get_area(treshold, neg_dist_max, neg_dist_mean)
    FP = get_area(neg_dist_min, treshold, neg_dist_mean)

  # false and positive rates
  fpr = FP / (FP + TN + 1e-5)
  tpr = TP / (TP + FN + 1e-5)
  return fpr, tpr

def plot_roc_curve(pos_dist_mean, neg_dist_mean, ax):
  treshold = np.linspace(0, 1)
  fpr, tpr = get_positive_rates(pos_dist_mean, neg_dist_mean, treshold)
  
  ax.set_title("ROC Curve")
  ax.plot([0, 1], [0, 1], "k--")
  ax.plot(fpr, tpr, "r")