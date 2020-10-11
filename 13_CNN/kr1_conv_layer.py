# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:31:40 2020

@author: 1052668570
"""
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

from keras.layers import Conv2D
from keras.models import Sequential
from scipy import misc

img = misc.ascent()
plt.imshow(img, cmap='gray')

img.shape
 
# =============================================================================
# Reshaping the image
# =============================================================================
img_tensor = img.reshape((1, 512, 512, 1))  # sample, height, width, channel

# =============================================================================
# Setting up the convolutional model but not trained
# =============================================================================
model = Sequential()
# starting with a 2D convolution of the image
# it's set up to take in 1 node a 3x3 window, or "filter"
model.add(Conv2D(1, kernel_size=(3, 3),
                 strides=(2, 1),
                 input_shape=(512, 512, 1),
                 padding='same'))
model.compile(optimizer='adam',
              loss='mse')

img_pred_tensor = model.predict(img_tensor)
img_pred_tensor.shape

img_pred = img_pred_tensor[0, :, :, 0]
plt.imshow(img_pred, cmap='gray')

# =============================================================================
# Extracting the inferred filter
# =============================================================================
weights = model.get_weights()
weights[0].shape  # Height, Width, channel input, channel output
plt.imshow(weights[0][:, :, 0, 0], cmap='gray')

# =============================================================================
# New filter
# =============================================================================
weights[0] = np.ones(weights[0].shape)
plt.imshow(weights[0][:, :, 0, 0])

img_pred_tensor = model.predict(img_tensor)
img_pred_tensor.shape

img_pred = img_pred_tensor[0, :, :, 0]
plt.imshow(img_pred, cmap='gray')