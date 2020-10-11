# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:32:13 2020

@author: 1052668570
Single multi-layer precepton / neural network

NOTE: HERE IN THE DERIVATES WE USE (targets(y_true) - y_pred), if we use this
format, then in the update weights we need to sum:
            # Gradient descent and update paremeters
        W += lr*(gradient_descent_W(outputs, y_train_enc, X_train) - reg*W)
        b += lr*(gradient_descent_b(outputs, y_train_enc) - reg*b)

    BUT, WE CAN USE TOO (y_pred - targets(y_true)), in this case, instead of add
    the gradient, we need to substract it sucha that:

            # Gradient descent and update paremeters
        W -= lr*(gradient_descent_W(outputs, y_train_enc, X_train) - reg*W)
        b -= lr*(gradient_descent_b(outputs, y_train_enc) - reg*b)


"""
import numpy as np


def forward(X, W1, b1, W2, b2, activation):
    if activation == 'sigmoid':
        # first layer
        a = X.dot(W1) + b1
        Z = 1 / (1 + np.exp(-a))

        # output layer
        a = Z.dot(W2) + b2
        # Softwmax
        outputs = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)

        return outputs, Z

    elif activation == 'relu':
        # first layer
        a = X.dot(W1) + b1
        Z = a * (a > 0)

        # output layer
        a = Z.dot(W2) + b2
        # Softwmax
        outputs = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)
        return outputs, Z


def gradient_descent_W2(y_pred, targets, Z):
    return Z.T.dot(targets - y_pred)


def gradient_descent_b2(y_pred, targets):
    return (targets - y_pred).sum(axis=0)


def gradient_descent_W1(y_pred, targets, Z, X, W2, activation):
    if activation == 'sigmoid':
        dw1 = (targets - y_pred).dot(W2.T) * (Z * (1 - Z))
        return X.T.dot(dw1)

    elif activation == 'relu':
        dw1 = (targets - y_pred).dot(W2.T) * (Z > 0)
        return X.T.dot(dw1)


def gradient_descent_b1(y_pred, targets, Z, W2, activation):
    if activation == 'sigmoid':
        return ((targets - y_pred).dot(W2.T) * (Z * (1 - Z))).sum(axis=0)

    elif activation == 'relu':
        return ((targets - y_pred).dot(W2.T) * (Z > 0)).sum(axis=0)
