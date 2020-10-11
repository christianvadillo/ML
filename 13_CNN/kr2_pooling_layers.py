# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:25:26 2020

@author: 1052668570
"""
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import MaxPooling2D, AveragePooling2D, AvgPool2D

from scipy import misc

img = misc.ascent()

plt.imshow(img, cmap='gray')
img.shape
# =============================================================================
# Reshaping the image
# =============================================================================
img_tensor = img.reshape(1, 512, 512, 1)


# =============================================================================
# MaxPooling layer
# =============================================================================
model = Sequential()
model.add(MaxPooling2D(pool_size=(5, 5), input_shape=(512, 512, 1)))
model.compile(optimizer='adam', loss='mse')

img_pred = model.predict(img_tensor)
img_pred.shape
plt.imshow(img_pred[0, :, :, 0], cmap='gray')

# =============================================================================
# AveragePooling2D layer
# =============================================================================
model = Sequential()
model.add(AveragePooling2D(pool_size=(5, 5), input_shape=(512, 512, 1)))
model.compile(optimizer='adam', loss='mse')

img_pred_avg = model.predict(img_tensor)
img_pred_avg.shape
plt.figure()
plt.imshow(img_pred_avg[0, :, :, 0], cmap='gray')

# =============================================================================
# AvgPool2D layer
# =============================================================================
model = Sequential()
model.add(AvgPool2D(pool_size=(5, 5), input_shape=(512, 512, 1)))
model.compile(optimizer='adam', loss='mse')

img_pred_avg2 = model.predict(img_tensor)
img_pred_avg2.shape
plt.figure()
plt.imshow(img_pred_avg2[0, :, :, 0], cmap='gray')
