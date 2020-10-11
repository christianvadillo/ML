# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:11:53 2020

@author: 1052668570
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from scipy import misc

img = misc.ascent()
plt.imshow(img)

# =============================================================================
# Creating the kernels (filter to apply)
# =============================================================================
# Horizontal filter
h_kernel = np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])

plt.imshow(h_kernel)

# =============================================================================
# Applying the filter
# =============================================================================
res = convolve2d(img, h_kernel)
plt.imshow(res)

# =============================================================================
# Creating the kernels (filter to apply)
# =============================================================================
# vertical filter
h_kernel = np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]]).T

plt.imshow(h_kernel)

# =============================================================================
# Applying the filter
# =============================================================================
res2 = convolve2d(img, h_kernel)
plt.figure()
plt.imshow(res2)
