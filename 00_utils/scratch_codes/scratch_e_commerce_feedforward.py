# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:57:37 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('whitegrid')
data = pd.read_csv('../data/classification/ecommerce_data.csv')

data.head(1).T
data.info()
data.describe()

sns.pairplot(data)

# =============================================================================
# Exploratory
# =============================================================================
# Mising values?
sns.heatmap(data.isna(), cmap='viridis', cbar=False, yticklabels=False)
data.isna().sum()

for i, n in enumerate(data.columns):
    plt.subplot(2, 3, i+1)
    plt.hist(data[n], bins=30)
    plt.title(n)
plt.tight_layout()

# =============================================================================
# Splitting data
# =============================================================================
# shuffle it
data = data.sample(frac=1)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

N, D = X.shape  # Samples, Dimmensions

# =============================================================================
# Scratch One-Hot Encoding for 'time_of_day'
# =============================================================================
X2 = np.zeros((N, D+3))  # New array with 4 extra columns (one for each category)
X2[:, 0:(D-1)] = X[:, 0:(D-1)] # Transfering values from X to X2 without 'time_of_day'

# One-hot encoding
for i, class_ in enumerate(X[:, -1]):
    X2[i, int(4+class_)] = 1

# assign X2 back to X, since we don't need original anymore
X = X2

# =============================================================================
# Splitting data (X_train, x_test)
# =============================================================================
X_train = X[:-100]
X_test = X[-100:]
y_train = y[:-100]
y_test = y[-100:]

N, D = X_train.shape  # Updating N, D with X_train shape
# =============================================================================
# Scaling X_1, X_2
# =============================================================================
# normalize columns 1 and 2
for i in (1, 2):
    mean = X_train[:, i].mean()
    std = X_train[:, i].std()
    X_train[:, i] = (X_train[:, i] - mean) / std
    X_test[:, i] = (X_test[:, i] - mean) / std

# =============================================================================
# Initilizing the model
# =============================================================================
M = 5
K = len(set(y))

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

# feedforward (X_train)
z = np.tanh(X_train.dot(W1) + b1)
# softmax
a = z.dot(W2) + b2
output = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)


# Predictions
y_pred = np.argmax(output, axis=1)

# classification_rate
print(f'Score: {np.mean(y_train == y_pred)}')

# # =============================================================================
# # Scratch One-Hot Encoding Y labels
# # =============================================================================
# y_cat = np.zeros((len(y), 4))

# for i, class_ in enumerate(y):
#     y_cat[i, class_] = 1


