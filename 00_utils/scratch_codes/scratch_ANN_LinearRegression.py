# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:04:26 2020

@author: 1052668570
Scratch:
    Linear Regression using ANN
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generate and plot the data
N = 500
X = np.random.random((N, 2)) * 4 - 2  # in between (-2, +2)
y = X[:, 0] * X[:, 1]  # makes a saddle shape


# Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], y)
# plt.show()

# =============================================================================
# # Setting up the model [ANN for regression]
# =============================================================================
D = 2  # Features
M = 100  # hidden units
train_losses = []
learning_rate = 0.0001
epochs = 200

# Layer 1
W = np.random.randn(D, M) / np.sqrt(D)  # Weight for layer 1
b = np.zeros(M)  # bias for layer 1

# Layer 2
V = np.random.randn(M) / np.sqrt(M)  # Weight for layer 2
c = 0  # bias for layer 2


# =============================================================================
# Main training loop
# =============================================================================
for epoch in range(epochs):
    # =============================================================================
    # Feedforward
    # =============================================================================
    # Layer 1
    Z = X.dot(W) + b
    Z = Z * (Z > 0)  # ReLU activation function
    y_hat = Z.dot(V) + c

    # =============================================================================
    # Calculating derivates for Gradient Descent
    # =============================================================================
    dV = (y - y_hat).dot(Z)
    dc = (y - y_hat).sum()
    dW = X.T.dot(np.outer(y - y_hat, V) * (Z > 0))  # ReLU activation function
    dz = np.outer(y - y_hat, V) * (Z > 0) # RelU activation function
    db = dz.sum(axis=0)  

    # =============================================================================
    # # Updating parameters with the gradients
    # =============================================================================
    V += learning_rate * dV
    c += learning_rate * dc
    W += learning_rate * dW
    b += learning_rate * db

    # =============================================================================
    # Get the loss
    # =============================================================================
    loss = ((y - y_hat) ** 2).mean()
    train_losses.append(loss)

    if epoch % 25 == 0:
        print(f'Epoch {epoch}/{epochs} - Train loss: {loss}')

# plt.plot(train_losses)
# plt.title('Train Loss')
# plt.xlabel('Epoch')
# plt.show()

# plot the prediction with the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)


# surface plot to compare the function that the NN learned vs
# the real data
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
X_grid = np.vstack((xx.flatten(), yy.flatten())).T


# Prediction
Z = X_grid.dot(W) + b
Z = Z * (Z > 0)  # ReLU activation function
Y_hat = Z.dot(V) + c

ax.plot_trisurf(X_grid[:, 0],
                X_grid[:, 1], 
                Y_hat, 
                linewidth=0.2, 
                antialiased=True,
                alpha=0.3)
plt.show()