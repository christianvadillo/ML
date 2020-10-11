# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:16:16 2020

@author: 1052668570

First neuronal network using tf 2.0
"""

import tensorflow as tf
import numpy as np

np.random.seed(101)
tf.random.set_seed(101)

# random_a = np.random.uniform(0, 100, (5, 5))
# random_b = np.random.uniform(0, 100, (5, 1))

# @tf.function
# def add_op(a, b):
#     return a + b

# @tf.function
# def multiply_op(a, b):
#     return a * b

# add_result = add_op(random_a, random_b)
# print(output)
# multiply_result = multiply_op(random_a, random_b)
# print(multiply_result)


# =============================================================================
# Basic ANN
# =============================================================================
n_features = 10
n_dense_neurons = 3  # layers

# Variables
x = np.random.random([1, n_features]).astype('float32')
W = tf.Variable(tf.random.normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))

# z = Wx + b
xW = tf.matmul(x, W)
z = tf.add(xW, b)
a = tf.nn.sigmoid(z)  # Activation function
print(a)

