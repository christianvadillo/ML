# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:13:01 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.cluster import KMeans

sns.set_style("whitegrid")

data = pd.read_csv(r'../data/clustering/College_Data', index_col=0)
data.info()
data.describe()

# =============================================================================
# Exploratory
# =============================================================================
sns.lmplot(x='Room.Board', y='Grad.Rate', data=data, hue='Private',
           fit_reg=False, palette='coolwarm', size=6, aspect=1)

sns.lmplot(x='Outstate', y='F.Undergrad', data=data, hue='Private',
           fit_reg=False, palette='coolwarm', size=6, aspect=1)

g = sns.FacetGrid(data, hue='Private', palette='coolwarm',size=6, aspect=1)
g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)

g = sns.FacetGrid(data, hue='Private', palette='coolwarm',size=6, aspect=1)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)

# =============================================================================
# Dealing with outlier in Grad.Rate
# =============================================================================
data[data['Grad.Rate'] > 100] = 100

g = sns.FacetGrid(data, hue='Private', palette='coolwarm',size=6, aspect=1)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)


# =============================================================================
# Kmeans model with 4 clusters
# =============================================================================
km = KMeans(n_clusters=2)
km.fit(data.drop(['Private'], axis=1))

# =============================================================================
# Evaluating model
# =============================================================================
def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0
    
data['cluster'] = data['Private'].apply(converter)

km.cluster_centers_
km.labels_

# Just if we have true labels
print(confusion_matrix(data['cluster'], km.labels_))
print(classification_report(data['cluster'], km.labels_))

