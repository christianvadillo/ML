# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:25:31 2020

@author: 1052668570
Testing regular momentum and nesterov momentum
"""

import numpy as np
import matplotlib.pyplot as plt

import scratch_mlp as mlp
import scratch_mnist_betchmark_utils as utls


def main(epochs=20, batch_size=500, activation='sigmoid', lr=0.0001):
    """ Compares 3 scenarios:
            1. Using batch SGD
            2. Using batch SGD with momentum
            3. Using batch SGD with Nesterov momentum
            4. Using batch SGD with RMSProp
    """
    print(f"Running test with {epochs} epochs and batch size of {batch_size}")
    # Load data
    X_train, X_test, y_train, y_test = utls.get_normalized_data()


    # Transform y-labels
    y_train_cat = utls.one_hot_encoding_labels(y_train)
    y_test_cat = utls.one_hot_encoding_labels(y_test)

    # Settings
    N, D = X_train.shape
    print_period = 50
    reg = 0.01
    n_batches = N // batch_size

    M = 300  # Neurons
    K = len(set(y_train))  # total of classes

    # Setting up the model
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.random.randn(M)

    # Output layer
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.random.randn(K)

    # Save initial weights
    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()

    # =========================================================================
    #      1. Using batch SGD
    # =========================================================================
    losses_batch = []
    errors_batch = []
    for epoch in range(epochs):
        for batch in range(n_batches):
            # Slicing data of batch size
            # [0 - 500], [500, 1000],.. [40500 - 41000]
            start = batch * batch_size
            end = start + batch_size
            X_train_batch = X_train[start:end]
            y_train_batch = y_train_cat[start:end]

            # forward step
            outputs, Z = mlp.forward(X_train_batch, W1, b1, W2, b2, activation)

            # backward step
            # Calculate gradients
            gW2 = mlp.gradient_descent_W2(outputs, y_train_batch, Z) + reg * W2
            gb2 = mlp.gradient_descent_b2(outputs, y_train_batch) + reg * b2
            gW1 = mlp.gradient_descent_W1(outputs, y_train_batch,
                                          Z, X_train_batch, W2, 
                                          activation) + reg * W1
            gb1 = mlp.gradient_descent_b1(outputs, y_train_batch,
                                          Z, W2, activation) + reg * b1

            # Update weights
            W2 += lr * gW2
            b2 += lr * gb2
            W1 += lr * gW1
            b1 += lr * gb1

            # Calculate loss and print it each 50 batchs
            if batch % print_period == 0:
                outputs_test, _ = mlp.forward(X_test, W1, b1,
                                              W2, b2, activation)
                loss = utls.loss_function(outputs_test, y_test_cat)
                losses_batch.append(loss)
                print(f"Loss at iteration {epoch}, {batch}: {loss:.3f}")
            
                e = utls.error_rate(outputs_test, y_test)
                errors_batch.append(e)
                print("Error rate:", e)

    # Final error
    outputs_test, _ = mlp.forward(X_test, W1, b1,
                                              W2, b2, activation)
    print("Final Error rate:", utls.error_rate(outputs_test, y_test))
    
    
    # =========================================================================
    #      2. Using batch SGD with momentum
    # =========================================================================
    # Reset weight values with the defaults saved
    W2 = W2_0.copy()
    b2 = b2_0.copy()
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    
    losses_momentum = []
    errors_momentum = []
    
    # Settings for momentum
    mu = 0.9
    # Initial velocities
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0
    
    for epoch in range(epochs):
        for batch in range(n_batches):
            # Slicing data of batch size
            # [0 - 500], [500, 1000],.. [40500 - 41000]
            start = batch * batch_size
            end = start + batch_size
            
            X_train_batch = X_train[start:end]
            y_train_batch = y_train_cat[start:end]
            
            # forward step
            outputs, Z = mlp.forward(X_train_batch, W1, b1, W2, b2, activation)
            
            # backward step with momentum
            # Calculate gradients (as a normal way)
            gW2 = mlp.gradient_descent_W2(outputs, y_train_batch, Z) + reg * W2
            gb2 = mlp.gradient_descent_b2(outputs, y_train_batch) + reg * b2
            gW1 = mlp.gradient_descent_W1(outputs, y_train_batch,
                                          Z, X_train_batch, W2, 
                                          activation) + reg * W1
            gb1 = mlp.gradient_descent_b1(outputs, y_train_batch,
                                          Z, W2, activation) + reg * b1

            # Update weights [Using momentum]
             # Update velocities
            dW2 = mu * dW2 - lr * gW2
            db2 = mu * db2 - lr * gb2
            dW1 = mu * dW1 - lr * gW1
            db1 = mu * db1 - lr * gb1
            
             # Update the weights with momentum
            W2 -= dW2
            b2 -= db2
            W1 -= dW1
            b1 -= db1
            
            # Calculate loss and print it each 50 batchs
            if batch % print_period == 0:
                outputs_test, _ = mlp.forward(X_test, W1, b1,
                                              W2, b2, activation)
                loss = utls.loss_function(outputs_test, y_test_cat)
                losses_momentum.append(loss)
                print(f"Loss at iteration {epoch}, {batch}: {loss:.3f}")
            
                e = utls.error_rate(outputs_test, y_test)
                errors_momentum.append(e)
                print("Error rate:", e)

    # Final error
    outputs_test, _ = mlp.forward(X_test, W1, b1,
                                              W2, b2, activation)
    print("Final Error rate:", utls.error_rate(outputs_test, y_test))
            
    # =========================================================================
    #      3. Using batch SGD with Nesterov momentum
    # =========================================================================
    # Reset weight values with the defaults saved
    W2 = W2_0.copy()
    b2 = b2_0.copy()
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    
    losses_nesterov = []
    errors_nesterov = []
    
    # Settings for momentum
    mu = 0.9
    # Initial velocities
    vW2 = 0
    vb2 = 0
    vW1 = 0
    vb1 = 0
    
    for epoch in range(epochs):
        for batch in range(n_batches):
            # Slicing data of batch size
            # [0 - 500], [500, 1000],.. [40500 - 41000]
            start = batch * batch_size
            end = start + batch_size
            
            X_train_batch = X_train[start:end]
            y_train_batch = y_train_cat[start:end]
            
            # forward step
            outputs, Z = mlp.forward(X_train_batch, W1, b1, W2, b2, activation)
            
            # backward step with momentum
            # Calculate gradients (as a normal way)
            gW2 = mlp.gradient_descent_W2(outputs, y_train_batch, Z) + reg * W2
            gb2 = mlp.gradient_descent_b2(outputs, y_train_batch) + reg * b2
            gW1 = mlp.gradient_descent_W1(outputs, y_train_batch,
                                          Z, X_train_batch, W2, 
                                          activation) + reg * W1
            gb1 = mlp.gradient_descent_b1(outputs, y_train_batch,
                                          Z, W2, activation) + reg * b1

            # Update weights [Using nesterov - momentum]
             # Update velocities
            vW2 = mu * vW2 - lr * gW2
            vb2 = mu * vb2 - lr * gb2
            vW1 = mu * vW1 - lr * gW1
            vb1 = mu * vb1 - lr * gb1
            
             # Update the weights with nesterov
            W2 -= mu * vW2 - lr * gW2
            b2 -= mu * vb2 - lr * gb2
            W1 -= mu * vW1 - lr * gW1
            b1 -= mu * vb1 - lr * gb1
            
            # Calculate loss and print it each 50 batchs
            if batch % print_period == 0:
                outputs_test, _ = mlp.forward(X_test, W1, b1,
                                              W2, b2, activation)
                loss = utls.loss_function(outputs_test, y_test_cat)
                losses_nesterov.append(loss)
                print(f"Loss at iteration {epoch}, {batch}: {loss:.3f}")
            
                e = utls.error_rate(outputs_test, y_test)
                errors_nesterov.append(e)
                print("Error rate:", e)

    # Final error
    outputs_test, _ = mlp.forward(X_test, W1, b1,
                                              W2, b2, activation)
    print("Final Error rate:", utls.error_rate(outputs_test, y_test))

    # =========================================================================
    #      3. Using batch RMSProp
    # =========================================================================
    # Reset weight values with the defaults saved
    W2 = W2_0.copy()
    b2 = b2_0.copy()
    W1 = W1_0.copy()
    b1 = b1_0.copy()

    losses_rmsprop = []
    errors_rmsprop = []

    # Settings for RMSprop
    decay_rate = 0.999
    eps = 1e-8
    # Initial caches for each parameter
    cache_W2 = 1
    cache_b2 = 1
    cache_W1 = 1
    cache_b1 = 1

    for epoch in range(epochs):
        for batch in range(n_batches):
            # Slicing data of batch size
            # [0 - 500], [500, 1000],.. [40500 - 41000]
            start = batch * batch_size
            end = start + batch_size

            X_train_batch = X_train[start:end]
            y_train_batch = y_train_cat[start:end]

            # forward step
            outputs, Z = mlp.forward(X_train_batch, W1, b1, W2, b2, activation)

            # backward step with rmsprop
            # Calculate gradients (as a normal way)
            gW2 = mlp.gradient_descent_W2(outputs, y_train_batch, Z) + reg * W2
            gb2 = mlp.gradient_descent_b2(outputs, y_train_batch) + reg * b2
            gW1 = mlp.gradient_descent_W1(outputs, y_train_batch,
                                          Z, X_train_batch, W2,
                                          activation) + reg * W1
            gb1 = mlp.gradient_descent_b1(outputs, y_train_batch,
                                          Z, W2, activation) + reg * b1

            # Update weights [Using RMSprop]
            # Update caches
            cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2*gW2
            cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2*gb2
            cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1*gW1
            cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1*gb1

            # Update the weights with RMSProp
            W2 += lr * gW2 / (np.sqrt(cache_W2) + eps)
            b2 += lr * gb2 / (np.sqrt(cache_b2) + eps)
            W1 += lr * gW1 / (np.sqrt(cache_W1) + eps)
            b1 += lr * gb1 / (np.sqrt(cache_b1) + eps)

            # Calculate loss and print it each 50 batchs
            if batch % print_period == 0:
                outputs_test, _ = mlp.forward(X_test, W1, b1,
                                              W2, b2, activation)
                loss = utls.loss_function(outputs_test, y_test_cat)
                losses_rmsprop.append(loss)
                print(f"Loss at iteration {epoch}, {batch}: {loss:.3f}")

                e = utls.error_rate(outputs_test, y_test)
                errors_rmsprop.append(e)
                print("Error rate:", e)

    # Final error
    outputs_test, _ = mlp.forward(X_test, W1, b1,
                                              W2, b2, activation)
    print("Final Error rate:", utls.error_rate(outputs_test, y_test))
    
    # =========================================================================
    #      3. Using batch Adam
    # =========================================================================
    # Reset weight values with the defaults saved
    W2 = W2_0.copy()
    b2 = b2_0.copy()
    W1 = W1_0.copy()
    b1 = b1_0.copy()

    losses_adam = []
    errors_adam = []

    # Settings for Adam
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    t = 1

    # Initial first moment
    mW2 = 0
    mb2 = 0
    mW1 = 0
    mb1 = 0

    # Initial second moment
    vW2 = 0
    vb2 = 0
    vW1 = 0
    vb1 = 0

    for epoch in range(epochs):
        for batch in range(n_batches):
            # Slicing data of batch size
            # [0 - 500], [500, 1000],.. [40500 - 41000]
            start = batch * batch_size
            end = start + batch_size

            X_train_batch = X_train[start:end]
            y_train_batch = y_train_cat[start:end]

            # forward step
            outputs, Z = mlp.forward(X_train_batch, W1, b1, W2, b2, activation)

            # backward step with rmsprop
            # Calculate gradients (as a normal way)
            gW2 = mlp.gradient_descent_W2(outputs, y_train_batch, Z) + reg * W2
            gb2 = mlp.gradient_descent_b2(outputs, y_train_batch) + reg * b2
            gW1 = mlp.gradient_descent_W1(outputs, y_train_batch,
                                          Z, X_train_batch, W2,
                                          activation) + reg * W1
            gb1 = mlp.gradient_descent_b1(outputs, y_train_batch,
                                          Z, W2, activation) + reg * b1

            # Update weights [Using Adam]
            # Update m
            mW2 = beta1 * mW2 + (1 - beta1) * gW2
            mb2 = beta1 * mb2 + (1 - beta1) * gb2
            mW1 = beta1 * mW1 + (1 - beta1) * gW1
            mb1 = beta1 * mb1 + (1 - beta1) * gb1

            # Update v
            vW2 = beta2 * vW2 + (1 - beta2) * gW2 * gW2
            vb2 = beta2 * vb2 + (1 - beta2) * gb2 * gb2
            vW1 = beta2 * vW1 + (1 - beta2) * gW1 * gW1
            vb1 = beta2 * vb1 + (1 - beta2) * gb1 * gb1

            # bias correction
            correction1 = 1 - beta1 ** t
            hat_mW2 = mW2 / correction1
            hat_mb2 = mb2 / correction1
            hat_mW1 = mW1 / correction1
            hat_mb1 = mb1 / correction1
            
            correction2 = 1 - beta2 ** t
            hat_vW2 = vW2 / correction2
            hat_vb2 = vb2 / correction2
            hat_vW1 = vW1 / correction2
            hat_vb1 = vb1 / correction2
            
            # update t
            t += 1

            # Apply updates to the params
            W2 = W2 + lr * hat_mW2 / np.sqrt(hat_vW2 + eps)
            b2 = b2 + lr * hat_mb2 / np.sqrt(hat_vb2 + eps)
            W1 = W1 + lr * hat_mW1 / np.sqrt(hat_vW1 + eps)
            b1 = b1 + lr * hat_mb1 / np.sqrt(hat_vb1 + eps)

            # Calculate loss and print it each 50 batchs
            if batch % print_period == 0:
                outputs_test, _ = mlp.forward(X_test, W1, b1,
                                              W2, b2, activation)
                loss = utls.loss_function(outputs_test, y_test_cat)
                losses_adam.append(loss)
                print(f"Loss at iteration {epoch}, {batch}: {loss:.3f}")

                e = utls.error_rate(outputs_test, y_test)
                errors_adam.append(e)
                print("Error rate:", e)

    # Final error
    outputs_test, _ = mlp.forward(X_test, W1, b1,
                                              W2, b2, activation)
    print("Final Error rate:", utls.error_rate(outputs_test, y_test))

    plt.figure()
    plt.plot(losses_batch, label='Train Loss [Batch]')
    plt.plot(losses_momentum, label='Train Loss [Batch-Momentum]')
    plt.plot(losses_nesterov, label='Train Loss [Batch-Nesterov]')
    plt.plot(losses_rmsprop, label='Train Loss [Batch-RMSProp]')
    plt.plot(losses_adam, label='Train Loss [Batch-Adam]')
    plt.legend()

    plt.figure()
    plt.plot(errors_batch, label='Train Error [Batch]')
    plt.plot(errors_momentum, label='Train Error [Batch-Momentum]')
    plt.plot(errors_nesterov, label='Train Error [Batch-Nesterov]')
    plt.plot(errors_rmsprop, label='Train Error [Batch-RMSProp]')
    plt.plot(errors_adam, label='Train Error [Batch-Adam]')
    plt.legend()


main(epochs=20, batch_size=500, activation='relu')
