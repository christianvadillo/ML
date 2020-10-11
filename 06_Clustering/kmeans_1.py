# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:51:50 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

sns.set_style("whitegrid")

# =============================================================================
# Generating artificial data
# =============================================================================
data = make_blobs(n_samples=200, n_features=2, centers=4, 
                  cluster_std=1.8, random_state=101)

data[0].shape  # data
data[1].shape  # clusters
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')


# =============================================================================
# Kmeans model with 4 clusters
# =============================================================================
km = KMeans(n_clusters=4)
km.fit(data[0])

# =============================================================================
# Evaluating model
# =============================================================================
km.cluster_centers_
km.labels_

fig, (ax1, ax2)= plt.subplots(1,2, sharey=True, figsize=(10,6))
ax1.set_title('K means')
ax1.scatter(data[0][:,0], data[0][:,1], c=km.labels_, cmap='rainbow')
ax2.set_title('True')
ax2.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')

