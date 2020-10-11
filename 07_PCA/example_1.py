# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:16:23 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

cancer = load_breast_cancer()

cancer.keys()
print(cancer['DESCR'])

data = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])

# =============================================================================
# # Exploratory
# =============================================================================

data.head(1).T
data.info()
data.describe()

# =============================================================================
# # Scaling data
# =============================================================================
sc = StandardScaler()
scaled_data = sc.fit_transform(data)

# =============================================================================
# # Model - Using PCA to visualize 2-D space
# =============================================================================
pca = PCA(n_components=2)
pca.fit(scaled_data)

# =============================================================================
# # Evaluation
# =============================================================================
pca.explained_variance_
pca.explained_variance_ratio_

x_pca = pca.transform(scaled_data)


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'], cmap='plasma')
plt.xlabel("PC1")
plt.ylabel("PC2")


# =============================================================================
# Graph
# =============================================================================
pca.components_
components = pd.DataFrame(pca.components_, columns=cancer['feature_names'])
components  # Each column represent the weights for each PC

plt.figure(figsize=(12,6))
sns.heatmap(components, cmap='plasma')
plt.tight_layout()
