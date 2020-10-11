# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:12:37 2020

@author: 1052668570
"""

import numpy as np

# Neural network output for one sample
a = np.random.randn(5)

# =============================================================================
# softmax for one sample
# softmax(a) = e^a / sum(e^a)
# =============================================================================
expa = np.exp(a)
expa = expa / expa.sum()

print(expa)
print(expa.sum())

# =============================================================================
# softmax for many samples at inputs
# softmax(a) = e^a / sum(e^a)
# =============================================================================
expa = np.exp(a)
expa = expa / expa.sum()

print(expa)
print(expa.sum())

# Neural network output for many samples
A = np.random.randn(100, 5)

# =============================================================================
# softmax for one sample
# softmax(a) = e^a / sum(e^a)
# =============================================================================
expA = np.exp(A)
answer = expa / expa.sum(axis=1, keepdims=True)  # divide each row

print(answer)
print(answer.sum(axis=1))  # sum around rows

# =============================================================================
# softmax for many samples at inputs
# softmax(a) = e^a / sum(e^a)
# =============================================================================
expa = np.exp(a)
expa = expa / expa.sum()

print(expa)
print(expa.sum())
