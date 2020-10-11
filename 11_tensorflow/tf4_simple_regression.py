# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:52:02 2020

@author: 1052668570

Creating ANN for simple linear regression

y = mx + b

Not working
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(101)
tf.random.set_seed(101)

# # ===========================================================================
# # Simple regression Example
# # ===========================================================================


class Model(object):
    def __init__(self):
        # Initilizing weights and bias
        self.M = tf.Variable(np.random.rand(1))
        self.B = tf.Variable(np.random.rand(1))

    def __call__(self, x):
        return self.M * x + self.B


# # Cost (Loss) function
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


# x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
# y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
# plt.plot(x_data, y_label, '*')
# x_data.shape

model = Model()
# model(1.5).numpy()

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random.normal(shape=[NUM_EXAMPLES], dtype='float64')
noise = tf.random.normal(shape=[NUM_EXAMPLES], dtype='float64')
outputs = inputs * TRUE_W + TRUE_b + noise

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()
print('Current loss: %1.6f' % loss(model(inputs), outputs).numpy())


# =============================================================================
# Training loop
# =============================================================================
def train(model, inputs, outputs, learning_rate):
    """
     train the model using gradient descent to update the weights 
     variable (M) and the bias variable (B) to reduce the loss

    """
    with tf.GradientTape() as t:
        current_loss = loss(outputs, model(inputs))
        dM, dB = t.gradient(current_loss, [model.M, model.B])
        model.M.assign_sub(learning_rate * dM)  # for decrementing a value
        model.B.assign_sub(learning_rate * dB)  # for decrementing a value


# =============================================================================
# Running the training
# =============================================================================
model = Model()
# Collect the history of M-values and B-values to plot later
Ms, Bs = [], []

epochs = range(100)
for epoch in epochs:
    Ms.append(model.M.numpy())
    Bs.append(model.B.numpy())
    current_loss = loss(outputs, model(inputs))

    train(model, inputs, outputs, 0.1)
    print('Epoch %2d: M=%1.2f B=%1.2f, loss=%2.5f' %
        (epoch, Ms[-1], Bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ms, 'r',
         epochs, Bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['M', 'B', 'True M', 'True B'])
plt.show()

