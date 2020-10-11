# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:54:39 2020

@author: 1052668570
"""

import numpy as np
import matplotlib.pyplot as plt

f = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype='float32')
g = np.array([-1, 1], dtype='float32')

convolution = np.convolve(f, g)

plt.subplot(211)
plt.plot(f, 'o-', label='f')
plt.legend()
plt.subplot(212)
plt.plot(convolution, 'ro-', label='convolution')
plt.legend()
