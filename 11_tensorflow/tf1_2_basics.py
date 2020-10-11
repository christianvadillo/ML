# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:08:45 2020

@author: 1052668570
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


def display_sample(num):
    # Print sample label
    label = y_train[num]

    # Reshape the 784 values to a 28*28 image
    image = X_train[num].reshape([28, 28])
    plt.title('Sample: %d Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))


def display_data(num):
    images = X_train[num].reshape([1, 784])
    for i in range(1, 500):
        images = np.concatenate((images, X_train[i].reshape([1, 784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()


# MNIST dataset parameters
num_classes = 10  # total classes (0-10 digits)
num_features = 784  # data features (img shape: 28 * 28)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Converting to float32
X_train = np.array(X_train, np.float32)
X_test = np.array(X_test, np.float32)

# Flatten images to 1-D vector of 784 features
X_train = X_train.reshape([-1, num_features])
X_test = X_test.reshape([-1, num_features])

# Normalize images values from [0, 255] to [0, 1]
X_train = X_train / 255.
X_test = X_test / 255.

# Visualizing
# display_sample(2110)
# # How data is being feed into the network
# display_data(2110)


# =============================================================================
# Training parameters
# =============================================================================
learning_rate = 0.001
training_steps = 3000
batch_size = 300
display_step = 100

# Network parameters
n_hidden = 512  # number of neurons

# =============================================================================
# using tf.data API to shuffle and batch data
# =============================================================================
# Creating tf Dataset object
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))

train_data = train_data.repeat().shuffle(60000).batch(batch_size).prefetch(1)

# =============================================================================
# Setting up ANN
# =============================================================================
# To randomly initialize the variable
random_normal = tf.initializers.RandomNormal()
# Creating 512 variables that contains weights for hidden neurons
weights = {'h': tf.Variable(random_normal([num_features, n_hidden])),
           'out': tf.Variable(random_normal([n_hidden, num_classes]))}

# Setting bias for the hidden layers
bias = {'b': tf.Variable(tf.zeros([n_hidden])),
        'out': tf.Variable(tf.zeros([num_classes]))}


# =============================================================================
# Creating the model
# =============================================================================
def neuronal_net(input_data):
    # Hidden fully connected layer with 512 neurons
    hidden_layer = tf.add(tf.matmul(input_data, weights['h']), bias['b'])
    # Apply sigmoid function to hidden layer output for non-linearity
    hidden_layer = tf.nn.sigmoid(hidden_layer)  # activation function

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(hidden_layer, weights['out'] + bias['out'])
    # Apply softmax to normalize the logits to a probability distribution
    return tf.nn.softmax(out_layer)


# =============================================================================
# Loss function
# =============================================================================
def cross_entropy(y_pred, y_true):
    # Encode label to one hot vector
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


# =============================================================================
# Setting up our stocastic gradient descent optimizer
# =============================================================================
optimizer = tf.keras.optimizers.SGD(learning_rate)


def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differntiation
    with tf.GradientTape() as g:
        pred = neuronal_net(x)
        loss = cross_entropy(pred, y)

    # Variables to update (trainable variables)
    trainable_variables = list(weights.values()) + list(bias.values())
    
    # Compute gradients
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# =============================================================================
# Setting up Accuracy metric
# =============================================================================
def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# =============================================================================
# Run training
# =============================================================================
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = neuronal_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("Training epoch: %i, Loss: %f, Accuracy: %f" % (step, loss, acc))
