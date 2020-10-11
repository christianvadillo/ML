# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:01:34 2020

@author: 1052668570

Contains bechmark for Mnist using PCA and Logistic Regression

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
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# df = pd.read_csv('D:/PYTHON_PROJECTS/ML/data/kaggle/digit-recognizer/train.csv')


def get_transformed_data():
    """
    This function returns data transformed using PCA
    and centered to mean 0

    Returns
    X: data transformed with PCA
    y: Targed data
    pca: PCA model used
    mu: mean used

    """
    print("Reading in and transforming digit-recognizer data...")
    df = pd.read_csv('D:/PYTHON_PROJECTS/ML/data/kaggle/digit-recognizer/train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)

    X = data[:, 1:]
    y = data[:, 0]
    y = y.astype(np.int32)

    # Splitting the data
    X_train = X[:-1000]
    y_train = y[:-1000]
    X_test = X[-1000:]
    y_test = y[-1000:]

    # center the data
    mu = X_train.mean(axis=0)
    X_train = X_train - mu
    X_test = X_test - mu

    # transform the data using PCA
    pca = PCA()
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)

    # plot_cumulative_variance(pca)

    # take first 300 cols of Z
    Z_train = Z_train[:, :300]
    Z_test = Z_test[:, :300]

    # normalize Z
    mu = Z_train.mean(axis=0)
    std = Z_train.std(axis=0)
    Z_train = (Z_train - mu) / std
    Z_test = (Z_test - mu) / std

    return Z_train, Z_test, y_train, y_test


def get_normalized_data():
    print("Reading in and transforming mnist data...")
    df = pd.read_csv('D:/PYTHON_PROJECTS/ML/data/kaggle/digit-recognizer/train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)

    X = data[:, 1:]
    y = data[:, 0]
    y = y.astype(np.int32)

    # Splitting the data
    X_train = X[:-1000]
    y_train = y[:-1000]
    X_test = X[-1000:]
    y_test = y[-1000:]

    mu = X_train.mean(axis=0)  # mean of each column
    std = X_train.std(axis=0)  # std of each column
    np.place(std, std == 0, 1)  # Changing std of 0 to 1
    X_train = (X_train - mu) / std  # normalize the data
    X_test = (X_test - mu) / std  # normalize the data
    return X_train, X_test, y_train, y_test


def plot_cumulative_variance(pca):
    pcs = []
    for pc in pca.explained_variance_ratio_:
        if len(pcs) == 0:
            pcs.append(pc)
        else:
            pcs.append(pc + pcs[-1])

    plt.plot(pcs)
    plt.ylabel('Cumulative variance')
    plt.xlabel("PCS")


def forward(X, W, b):
    a = X.dot(W) + b  # Nueron output
    # softmax
    outputs = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)

    return outputs


def predict(outputs):
    return np.argmax(outputs, axis=1)


def error_rate(outputs, targets):
    y_pred = predict(outputs)
    return np.mean(y_pred != targets)


def loss_function(y_pred, targets):
    """ Multiclass Cross-Entropy """
    loss = targets * np.log(y_pred)
    return -loss.sum()


def gradient_descent_W(y_pred, targets, X):
    return X.T.dot(targets - y_pred)


def gradient_descent_b(y_pred, targets):
    return (targets - y_pred).sum(axis=0)


def one_hot_encoding_labels(y):
    N = len(y)
    encoded = np.zeros((N, len(set(y))))
    for i in range(N):
        encoded[i, y[i]] = 1
    return encoded


def bechmark_full(lr=0.0004, epochs=500):
    X_train, X_test, y_train, y_test = get_normalized_data()
    print(" Performing logistic regression using normalized data...")

    # One hot encoding y labels
    N, D = X_train.shape
    y_train_enc = one_hot_encoding_labels(y_train)
    y_test_enc = one_hot_encoding_labels(y_test)

    # For save loss and rates
    losses_train = np.zeros(epochs)
    losses_test = np.zeros(epochs)
    classification_rates_test = np.zeros(epochs)

    # Initilize the parameters of the model
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    lr = lr
    reg = 0.01
    epochs = epochs

    # Main lopp train
    for epoch in range(epochs):
        # forward step train
        outputs = forward(X_train, W, b)
        loss = loss_function(outputs, y_train_enc)
        losses_train[epoch] = loss

        # forward step test
        outputs_test = forward(X_test, W, b)
        loss_test = loss_function(outputs_test, y_test_enc)
        losses_test[epoch] = loss_test

        # Classification error rate
        error = error_rate(outputs_test, y_test)
        classification_rates_test[epoch] = error

        # Gradient descent and update paremeters
        W += lr*(gradient_descent_W(outputs, y_train_enc, X_train) - reg*W)
        b += lr*(gradient_descent_b(outputs, y_train_enc) - reg*b)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} - Train Loss: {loss:.3f}, Test Loss {loss_test:.3f}, Error rate: {error:.3f}')

    # Final error
    outputs = forward(X_test, W, b)
    print(f'Final error rate: {error_rate(outputs, y_test)}')
    plt.figure()
    plt.plot(losses_train, label='Train Loss')
    plt.plot(losses_test, label='Test Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(classification_rates_test)
    plt.title("classification error rates")
    plt.show()


def bechmark_pca(lr=0.00004, epochs=100):
    X_train, X_test, y_train, y_test = get_transformed_data()
    print(" Performing logistic regression using pca transform...")

    # One hot encoding y labels
    N, D = X_train.shape
    y_train_enc = one_hot_encoding_labels(y_train)
    y_test_enc = one_hot_encoding_labels(y_test)

    # For save loss and rates
    losses_train = np.zeros(epochs)
    losses_test = np.zeros(epochs)
    classification_rates_test = np.zeros(epochs)

    # Initilize the parameters of the model
    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    lr = lr
    reg = 0.01
    epochs = epochs

    # Main lopp train
    for epoch in range(epochs):
        # forward step train
        outputs = forward(X_train, W, b)
        loss = loss_function(outputs, y_train_enc)
        losses_train[epoch] = loss

        # forward step test
        outputs_test = forward(X_test, W, b)
        loss_test = loss_function(outputs_test, y_test_enc)
        losses_test[epoch] = loss_test

        # Classification error rate
        error = error_rate(outputs_test, y_test)
        classification_rates_test[epoch] = error

        # Gradient descent and update paremeters
        W += lr*(gradient_descent_W(outputs, y_train_enc, X_train) - reg*W)
        b += lr*(gradient_descent_b(outputs, y_train_enc) - reg*b)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} - Train Loss: {loss:.3f}, Test Loss {loss_test:.3f}, Error rate: {error:.3f}')

    # Final error
    outputs = forward(X_test, W, b)
    print(f'Final error rate: {error_rate(outputs, y_test)}')
    plt.figure()
    plt.plot(losses_train, label='Train Loss')
    plt.plot(losses_test, label='Test Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(classification_rates_test)
    plt.title("classification error rates")
    plt.show()


# bechmark_pca()

    # N = len(y_train)
    # encoded = np.zeros((N, len(set(y_train))))
    # for i in range(N):
    #     encoded[i, int(y_train[i]]) = 1
    # return encoded