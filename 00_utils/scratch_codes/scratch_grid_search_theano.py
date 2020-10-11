# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:42:00 2020

@author: 1052668570

Scratch code of grid search to optimize the hyperparameters af a ANN model
of one layer built it in theano
only works with python < 3.6
"""

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def init_weights(M1, M2):
    return np.random.randn(M1, M2) * np.sqrt(2.0 / M1)


def get_spiral():
    """
    Returns: spiral data with noise
    -------
    X : np.array
        DESCRIPTION.
    y : np.array
        DESCRIPTION.

    """
    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi * i / 3.0
        end_angle = start_angle + np.pi / 2
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points

    # Convert into cartesian coordinates
    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])

    # inputs
    X = np.empty((600, 2))
    X[:, 0] = x1.flatten()
    X[:, 1] = x2.flatten()

    # add noise
    X += np.random.randn(600, 2) * 0.5

    # targets
    y = np.array([0]*100 + [1]*100 + [0]*100 + [1]*100 + [0]*100 + [1]*100)

    return X, y


class HiddenLayer(object):
    def __init__(self, M1, M2, f):
        self.M1 = M1  # input  hidden layer size (M1, M2)
        self.M2 = M2  # output hidden layer size (M1, M2)
        self.f = f  # activation function
        W = init_weights(M1, M2)
        b = np.zeros(M2)
        self.W = theano.shared(W)
        self.b = theano.shared(b)
        self.params = [self.W, self.b]

    def forward(self, X):
        if self.f == T.nnet.relu:
            # If activation is relu
            return self.f(X.dot(self.W) + self.b, alpha=0.1)
        return self.f(X.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, y,
            activation=T.nnet.relu,
            learning_rate=1e-3,
            mu=0.0,
            reg=0,
            epochs=100,
            batch_size=None,
            print_period=100,
            show_fig=True):
        X = X.astype(np.float32)
        y = y.astype(np.int32)

        # Initilize hidden layers
        N, D = X.shape
        self.layers = []
        M1 = D  # input layer
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, activation)
            self.layers.append(h)
            M1 = M2

        # Output layer [Final layer]
        K = len(set(y))  # num of classes
        h = HiddenLayer(M1, K, f=T.nnet.softmax)
        self.layers.append(h)

        if batch_size is None:
            # Full GD
            batch_size = N

        # Collect params for later use
        self.params = []
        for h in self.layers:
            self.params += h.params

        # for momentum
        dparams = [theano.shared(np.zeros_like(p.get_value()))
                   for p in self.params]

        # Setting up theano functions and variables
        th_X = T.matrix('X')
        th_y = T.ivector('y')
        outputs = self.forward(th_X)

        r_loss = reg * T.mean([(p*p).sum() for p in self.params])
        print(outputs)
        # cost = -T.mean(T.log(p_y_given_x[T.arange(thY.shape[0]), thY])) #+ rcost
        loss = -T.mean(T.log(outputs[T.arange(th_y.shape[0], th_y)])) + r_loss
        prediction = T.argmax(outputs, axis=1)
        grads = T.grad(loss, self.params)

        # momentum only
        updates = [
            (p, p + mu*dp - learning_rate*g)
            for p, dp, g in zip(self.params, dparams, grads)
            ] + [
                (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
                ]

        train_op = theano.function(
            inputs=[th_X, th_y],
            outputs=[loss, prediction],
            updates=updates,
            )

        self.predict_op = theano.function(
            inputs=[th_X],
            outputs=prediction,
            )

        n_batches = N // batch_size
        losses = []
        for epoch in range(epochs):
            if n_batches > 1:
                X, y = shuffle(X, y)
            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                loss, outputs = train_op(X_batch, y_batch)
                losses.append(loss)
                if (batch+1) % print_period == 0:
                    print(f"epoch: {epoch+1}, batch: {batch+1}/{n_batches}, Loss: {loss}")

        if show_fig:
            plt.plot(losses)
            plt.show()

    def forward(self, X):
        outputs = X
        for h in self.layers:
            outputs = h.forward(outputs)
        return outputs

    def score(self, X, y):
        y_pred = self.predict_op(X)
        return np.mean(y == y_pred)

    def predict(self, X):
        return self.predict_op(X)


def grid_search():
    X, y = get_spiral()
    X, y = shuffle(X, y)

    # Splitting the data with 70/30
    train_size = int(0.7 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Hyperparameters to try
    hidden_layer_sizes = [
        [300],
        [100, 100],
        [50, 50, 50],
        ]

    learning_rates = [1e-4, 1e-3, 1e-2]
    l2_penalties = [0., 0.1, 1.0]

    # Loop through all possible hyperparameter settings
    best_validation_rate = 0
    best_hls = None
    best_lr = None
    best_l2 = None
    for hls in hidden_layer_sizes:
        for lr in learning_rates:
            for l2 in l2_penalties:
                model = ANN(hls)
                model.fit(X_train, y_train,
                          learning_rate=lr,
                          reg=l2,
                          mu=0.99,
                          epochs=3000,
                          show_fig=False)

    train_accuracy = model.score(X_train, y_train)
    validation_accuracy = model.score(X_test, y_test)
    print(f"Train accuracy:{train_accuracy}, Test accuracy: {validation_accuracy}, \
          settings: {hls}, {lr}, {l2}")

    if validation_accuracy > best_validation_rate:
        best_validation_rate = validation_accuracy
        best_hls = hls
        best_lr = lr
        best_l2 = l2

    print("Best Test accuracy:", best_validation_rate)
    print("Best settings:")
    print("hidden_layer_sizes:", best_hls)
    print("learning_rate:", best_lr)
    print("l2:", best_l2)


grid_search()
    

