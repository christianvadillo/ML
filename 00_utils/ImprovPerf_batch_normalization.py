# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:25:23 2020

@author: 1052668570
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical

print(tf.__version__)

# Loading data
data = load_digits()
X, y = data.data, data.target
X.shape

# Visualizing
# for i in range(8):
#     # images of 8x8 pixels
#     plt.subplot(1, 8, i+1)
#     plt.imshow(X.reshape(-1, 8, 8)[i], cmap='gray')

# =============================================================================
# Transforming labels using one hot encoding
# =============================================================================
y_cat = to_categorical(y, num_classes=10)

# =============================================================================
# Splitting the data
# =============================================================================
X_train, X_test, y_train, y_test = \
    train_test_split(X, y_cat, test_size=0.3, random_state=101)


# =============================================================================
# Define a function with our model[Using Functional API]
# =============================================================================
def repeated_training(**kwargs):
    histories = []
    # early_stopping = EarlyStopping(monitor='loss', patience=1)

    for repeat in range(kwargs['repeats']):
        K.clear_session()
        # Input layer
        inputs = Input(kwargs['X_train'].shape[1:], )
        # First layer
        x = Dense(kwargs['units'],
                  kernel_initializer='normal',
                  activation=kwargs['activation'])(inputs)
        # Add Batch Normalization?
        if kwargs['do_bn']:
            x = BatchNormalization()(x)

        # Second layer
        x = Dense(kwargs['units'],
                  kernel_initializer='normal',
                  activation=kwargs['activation'])(x)
        # Add Batch Normalization?
        if kwargs['do_bn']:
            x = BatchNormalization()(x)

        # Third layer
        x = Dense(kwargs['units'],
                  kernel_initializer='normal',
                  activation=kwargs['activation'])(x)
        # Add Batch Normalization?
        if kwargs['do_bn']:
            x = BatchNormalization()(x)

        # output layer
        outputs = Dense(10, activation='softmax')(x)

        # Registring the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Fitting the model
        h = model.fit(kwargs['X_train'], kwargs['y_train'],
                      validation_data=(kwargs['X_test'], kwargs['y_test']),
                      epochs=kwargs['epochs'],
                      # callbacks=[early_stopping],
                      verbose=0)

        histories.append([h.history['accuracy'], h.history['val_accuracy']])
        print(repeat, end=' ')

    histories = np.array(histories)

    # calculate mean and standard deviation across repeats:
    mean_acc = histories.mean(axis=0)
    std_acc = histories.std(axis=0)
    print()

    return mean_acc[0], std_acc[0], mean_acc[1], std_acc[1]


# =============================================================================
# Compare models with batch normaliztion and without
# =============================================================================
mean_acc, std_acc, mean_acc_val, std_acc_val = \
    repeated_training(X_train=X_train,
                      y_train=y_train,
                      X_test=X_test,
                      y_test=y_test,
                      units=512,
                      epochs=10,
                      repeats=3,
                      activation='relu',
                      do_bn=False)


mean_acc_bn, std_acc_bn, mean_acc_val_bn, std_acc_val_bn = \
    repeated_training(X_train=X_train,
                      y_train=y_train,
                      X_test=X_test,
                      y_test=y_test,
                      units=512,
                      epochs=10,
                      repeats=3,
                      activation='relu',
                      do_bn=True)


# =============================================================================
# Plotting the learning curve
# =============================================================================
def plot_mean_std(m, s):
    plt.plot(m)
    plt.fill_between(range(len(m)), m-s, m+s, alpha=0.1)


plot_mean_std(mean_acc, std_acc)
plot_mean_std(mean_acc_val, std_acc_val)
plot_mean_std(mean_acc_bn, std_acc_bn)
plot_mean_std(mean_acc_val_bn, std_acc_val_bn)
# plt.ylim(0, 1.01)
plt.title('Batch Normalization Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test', 'Train whti Batch Normalization',
            'Test with Batch Normalization'])
