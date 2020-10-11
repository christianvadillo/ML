# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:03:14 2020

@author: 1052668570
"""

import numpy as np
import matplotlib.pyplot as plt
import scratch_mnist_betchmark_utils as utils

from sklearn.utils import shuffle
from datetime import datetime


def bechmark_full_gd(lr=0.0001, epochs=200):
    X_train, X_test, y_train, y_test = utils.get_transformed_data()
    print(" Performing logistic regression using pca transform...")

    # One hot encoding y labels
    N, D = X_train.shape
    y_train_enc = utils.one_hot_encoding_labels(y_train)
    y_test_enc = utils.one_hot_encoding_labels(y_test)

    # For save loss and rates
    losses_train = np.zeros(epochs)
    losses_test = np.zeros(epochs)
    classification_rates_test = np.zeros(epochs)

    # Initilize the parameters of the model
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    lr = lr
    reg = 0.01
    t0 = datetime.now()
    epochs = epochs

    # Main lopp train
    for epoch in range(epochs):
        # forward step train
        outputs = utils.forward(X_train, W, b)
        loss = utils.loss_function(outputs, y_train_enc)
        losses_train[epoch] = loss

        # forward step test
        outputs_test = utils.forward(X_test, W, b)
        loss_test = utils.loss_function(outputs_test, y_test_enc)
        losses_test[epoch] = loss_test

        # Classification error rate
        error = utils.error_rate(outputs_test, y_test)
        classification_rates_test[epoch] = error

        # Gradient descent and update paremeters
        W += lr*(utils.gradient_descent_W(outputs, y_train_enc, X_train) - reg*W)
        b += lr*(utils.gradient_descent_b(outputs, y_train_enc) - reg*b)

        # if epoch % 10 == 0:
        #     print(f'Epoch {epoch}/{epochs} - Train Loss: {loss:.3f}, Test Loss {loss_test:.3f}, Error rate: {error:.3f}')

    # Final error
    outputs = utils.forward(X_test, W, b)
    print(f'Final error rate: {utils.error_rate(outputs, y_test)}')
    print("Elapsed time for Full DD:", datetime.now() - t0)
    plt.figure()
    plt.plot(losses_train, label='Train Loss')
    plt.plot(losses_test, label='Test Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(classification_rates_test)
    plt.title("classification error rates")
    plt.show()


def bechmark_stochastic_gd(lr=0.0001, epochs=1):
    X_train, X_test, y_train, y_test = utils.get_transformed_data()
    print(" Performing logistic regression using pca transform...")

    # One hot encoding y labels
    N, D = X_train.shape
    y_train_enc = utils.one_hot_encoding_labels(y_train)
    y_test_enc = utils.one_hot_encoding_labels(y_test)

    # For save loss and rates
    losses_train = np.zeros(epochs)
    losses_test = np.zeros(epochs)
    classification_rates_test = np.zeros(epochs)

    # Initilize the parameters of the model
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    lr = lr
    reg = 0.01
    t0 = datetime.now()
    epochs = epochs

    # Main lopp train
    for epoch in range(epochs):
        tmpX, tmpY = shuffle(X_train, y_train_enc)
        for n in range(min(N, N)):  # reduced to 500 sample so it won't take so long...
            x = tmpX[n, :].reshape(1, D)
            y = tmpY[n, :].reshape(1, 10)

            # forward step train
            outputs = utils.forward(x, W, b)
            loss = utils.loss_function(outputs, y)
            losses_train[epoch] = loss

            # forward step test
            outputs_test = utils.forward(X_test, W, b)
            loss_test = utils.loss_function(outputs_test, y_test_enc)
            losses_test[epoch] = loss_test

            # Classification error rate
            error = utils.error_rate(outputs_test, y_test)
            classification_rates_test[epoch] = error

            # Gradient descent and update paremeters
            W += lr*(utils.gradient_descent_W(outputs, y, x) - reg*W)
            b += lr*(utils.gradient_descent_b(outputs, y) - reg*b)

            # if n % 10 == 0:
            #     print(f'Epoch {epoch}/{epochs} - Train Loss: {loss:.3f}, Test Loss {loss_test:.3f}, Error rate: {error:.3f}')

    # Final error
    outputs = utils.forward(X_test, W, b)
    print(f'Final error rate: {utils.error_rate(outputs, y_test)}')
    print("Elapsed time for Full DD:", datetime.now() - t0)
    plt.figure()
    plt.plot(losses_train, label='Train Loss')
    plt.plot(losses_test, label='Test Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(classification_rates_test)
    plt.title("classification error rates")
    plt.show()


def bechmark_batch_gd(lr=0.0001, epochs=200):
    X_train, X_test, y_train, y_test = utils.get_transformed_data()
    print(" Performing logistic regression using pca transform...")

    # One hot encoding y labels
    N, D = X_train.shape
    y_train_enc = utils.one_hot_encoding_labels(y_train)
    y_test_enc = utils.one_hot_encoding_labels(y_test)

    # For save loss and rates
    losses_train = np.zeros(epochs)
    losses_test = np.zeros(epochs)
    classification_rates_test = np.zeros(epochs)

    # Initilize the parameters of the model
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    lr = lr
    reg = 0.01
    t0 = datetime.now()
    epochs = epochs
    batch_size = 500
    n_batches = int(N / batch_size)

    # Main lopp train
    for epoch in range(epochs):
        tmpX, tmpY = shuffle(X_train, y_train_enc)
        for batch in range(n_batches):
            x = tmpX[batch*batch_size:(batch*batch_size+batch_size), :]
            y = tmpY[batch*batch_size:(batch*batch_size+batch_size), :]

            # forward step train
            outputs = utils.forward(x, W, b)
            loss = utils.loss_function(outputs, y)
            losses_train[epoch] = loss

            # forward step test
            outputs_test = utils.forward(X_test, W, b)
            loss_test = utils.loss_function(outputs_test, y_test_enc)
            losses_test[epoch] = loss_test

            # Classification error rate
            error = utils.error_rate(outputs_test, y_test)
            classification_rates_test[epoch] = error

            # Gradient descent and update paremeters
            W += lr*(utils.gradient_descent_W(outputs, y, x) - reg*W)
            b += lr*(utils.gradient_descent_b(outputs, y) - reg*b)

            # if batch % 10 == 0:
            #     print(f'Epoch {epoch}/{epochs} - Train Loss: {loss:.3f}, Test Loss {loss_test:.3f}, Error rate: {error:.3f}')

    # Final error
    outputs = utils.forward(X_test, W, b)
    print(f'Final error rate: {utils.error_rate(outputs, y_test)}')
    print("Elapsed time for Full DD:", datetime.now() - t0)
    plt.figure()
    plt.plot(losses_train, label='Train Loss')
    plt.plot(losses_test, label='Test Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(classification_rates_test)
    plt.title("classification error rates")
    plt.show()



bechmark_full_gd()
bechmark_stochastic_gd()
bechmark_batch_gd()