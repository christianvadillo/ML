# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:32:10 2020

@author: 1052668570
"""

import numpy as np
import matplotlib.pyplot as plt

Nclass = 500

# Gaussina clouds (inputs)
X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, -2])
X = np.vstack([X1, X2, X3])

# Labels
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

plt.scatter(X[:, 0], X[:, 1], c=Y, s=80, alpha=0.3)
plt.show()

D = 2  # features
M = 3  # Hidden layers
K = 3  # classes

# Initilizing Weigths
W1 = np.random.randn(D, M)
b1 = np.random.rand(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)


# feedforward function
def feedforward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1)-b1))  # sigmoid return of hidden layer
    A = Z.dot(W2) + b2  # second layer
    Y = np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)  # softmax
    return Y


# classification rate
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


# =============================================================================
# Applying
# =============================================================================
P_Y_given_X = feedforward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

assert(len(P) == len(Y))

print(f"Classification rate for randomly chosen weights:{classification_rate(Y, P)}")
