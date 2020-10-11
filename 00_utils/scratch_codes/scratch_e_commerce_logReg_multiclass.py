# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:57:37 2020

@author: 1052668570

Scratch:
    Logistic regression using softmax for multiclass on e-comerce data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle


sns.set_style('whitegrid')
data = pd.read_csv('../data/classification/ecommerce_data.csv')

data.head(1).T
data.info()
data.describe()

# sns.pairplot(data)

# =============================================================================
# Exploratory
# =============================================================================
# Mising values?
# sns.heatmap(data.isna(), cmap='viridis', cbar=False, yticklabels=False)
# data.isna().sum()

# for i, n in enumerate(data.columns):
#     plt.subplot(2, 3, i+1)
#     plt.hist(data[n], bins=30)
#     plt.title(n)
# plt.tight_layout()

# =============================================================================
# Splitting data
# =============================================================================
# shuffle it
# data = data.sample(frac=1)
data = shuffle(data)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.astype(np.int32)

N, D = X.shape  # Samples, Dimmensions
K = len(set(y))

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
# Scratch One-Hot Encoding for 'y'
# =============================================================================
y_cat = np.zeros((len(y), K))

for i, class_ in enumerate(y):
    y_cat[i, class_] = 1

# =============================================================================
# Splitting data (X_train, x_test)
# =============================================================================
X_train = X[:-100]
y_train = y_cat[:-100]
X_test = X[-100:]
y_test = y_cat[-100:]

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
# Initilizing the logistic regression model
# for multiclass
# =============================================================================
W = np.random.randn(D, K)
b = np.zeros(K)

train_losses = []
test_losses = []
learning_rate = 0.001
epochs = 10000
y_train_classes = np.argmax(y_train, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
# =============================================================================
#  main loop for training the model
# =============================================================================
for epoch in range(epochs):
    # feedforward (X_train)
    a_train = X_train.dot(W) + b  # neuron output (Logistic Regression)
    # softmax to 'a_train'
    output_train = np.exp(a_train) / np.exp(a_train).sum(axis=1, keepdims=True)

    # fedforward (X_test)
    a_test = X_test.dot(W) + b  # neuron output (Logistic Regression)
    # softmax to 'a_test'
    output_test = np.exp(a_test) / np.exp(a_test).sum(axis=1, keepdims=True)

    # =============================================================================
    # # Predictions
    # =============================================================================
    y_pred_train = np.argmax(output_train, axis=1)
    y_pred_test = np.argmax(output_test, axis=1)

    # =============================================================================
    # Loss function [multiclass cross entropy]
    # =============================================================================
    loss_train = -np.mean(y_train * np.log(output_train))
    loss_test = -np.mean(y_test * np.log(output_test))
    train_losses.append(loss_train)
    test_losses.append(loss_test)

    # =============================================================================
    # Updating parameters using Gradient Descent
    # =============================================================================
    W -= learning_rate * X_train.T.dot(output_train - y_train)
    b -= learning_rate * (output_train - y_train).sum(axis=0)

    # =============================================================================
    # # classification_rate
    # =============================================================================
    if epoch % 1000 == 0:
        score_tr = np.mean(y_train_classes == y_pred_train)
        score_te = np.mean(y_test_classes == y_pred_test)
        print(f'Epoch {epoch}/{epochs}, Train Loss: {loss_train}, Train Score: {score_tr} - Test Loss {loss_test}, Test Score: {score_te}')


plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()