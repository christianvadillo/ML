# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:34:36 2020

@author: 1052668570
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


def relu(x):
    """ activation function"""
    return (x > 0) * x


def relu2deriv(output):
    """ returns 1 for input > 0;
        returns 0 otherwise
    """
    return output > 0


lr = 0.2
hidden_units = 4
epochs = 60

# Input data
streetligths = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])

# target data
walks_or_stop = np.array([[1, 1, 0, 0]]).T

# Weights randomly initialized
weights_0_1 = 2 * np.random.random((streetligths.shape[1], hidden_units)) - 1
weights_1_2 = 2 * np.random.random((hidden_units, 1)) - 1

errors = np.zeros(epochs)
# ANN with 3 layers
for epoch in range(epochs):
    layer_2_error = 0
    for i, sample in enumerate(streetligths):
        # forward step
        layer_0 = sample  # First layer recieve sample inputs
        layer_1 = relu(layer_0.dot(weights_0_1))
        layer_2 = layer_1.dot(weights_1_2)  # Output

        # calculate the error
        layer_2_error += (layer_2 - walks_or_stop[i])**2

        # Computes the deltas
        layer_2_delta = (walks_or_stop[i] - layer_2).reshape(-1, 1)
        # at layer_1 given the delta at layer_2
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        # uptate the weights
        weights_1_2 += lr * layer_1.reshape(-1, 1).dot(layer_2_delta)
        weights_0_1 += lr * layer_0.reshape(-1, 1).dot(layer_1_delta)

    errors[epoch] = np.mean(layer_2_error)
    if epoch % 10 == 9:
        print(f"Epoch: {epoch}/{epochs} - Error: {layer_2_error}")


plt.plot(errors)