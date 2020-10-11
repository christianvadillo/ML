# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:45:10 2020

@author: 1052668570

y = mx + b
m = 0.5
b = 5
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

# =============================================================================
# Creating dataset
# =============================================================================
x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))
y_true = (0.5 * x_data) + 5 + noise
x_df = pd.DataFrame(x_data, columns=['X Data'])
y_df = pd.DataFrame(y_true, columns=['Y'])

data = pd.concat([x_df, y_df], axis=1)
data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
 

# =============================================================================
# Creating batches of data
# =============================================================================
BATCH_SIZE = 8
NUM_EPOCHS = 5
BATCHES = 100



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
epochs = range(1000)


for epoch in range(1000):
    Ms.append(model.M.numpy())
    Bs.append(model.B.numpy())
    current_loss = loss(y_true, model(x_data))

    train(model, x_data, y_true, 0.01)
    print('Epoch %2d: M=%1.2f B=%1.2f, loss=%2.5f' % 
          (epoch, Ms[-1], Bs[-1], current_loss))

# Let's plot it all
TRUE_M = 0.5
True_B = 5

plt.plot(epochs, Ms, 'r',
         epochs, Bs, 'b')
plt.plot([TRUE_M] * len(epochs), 'r--',
         [True_B] * len(epochs), 'b--')
plt.legend(['M', 'B', 'True M', 'True B'])
plt.show()


# # =============================================================================
# # Using Estimator API
# # NOT WORKING
# # =============================================================================
# from sklearn.model_selection import train_test_split

# # Defining feature column as numeric column
# feature_columns = [tf.feature_column.numeric_column('x', shape=(1,))]
# # Defining the estimator
# estimator = tf.estimator.LinearRegressor(feature_columns)

# X_train, y_train, x_test, y_test = train_test_split(x_data, y_true, test_size=0.3, random_state=101)

# # To defining batch and feed dictionary
# def input_fn():
#     BATCH_SIZE = 8
    

# input_function = tf.estimator.inputs.numpy_input_fn({'x': X_train },
#                                                     y_train,
#                                                     batch_size=8,
#                                                     num_epochs=None,
#                                                     shuffle=True
#                                                     )