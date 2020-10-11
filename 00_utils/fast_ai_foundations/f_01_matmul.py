# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:17:03 2020

@author: 1052668570
"""

import torch, matplotlib as mpl
import matplotlib.pyplot as plt
from util_functions import get_data

mpl.rcParams['image.cmap'] = 'gray'

X_train, y_train, X_test, y_test = get_data()

# Shape
n, c = X_train.shape
o = y_train.unique()
print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)
print(X_test)
print(X_test.shape)
print(y_test)
print(y_test.shape)
print(y_train.min(), y_train.max())

img = X_train[0]
img.view(28, 28).type
plt.imshow(img.view(28, 28))

# =============================================================================
# # INITIAL MODEL
# =============================================================================
weights = torch.randn(c, len(o))
print(weights.shape)
bias = torch.zeros(10)


# =============================================================================
# Matrix multiplication [Brute-Force][Slow]
# =============================================================================
def matmul_slow(a, b):
    ar, ac = a.shape  # rows, cols
    br, bc = b.shape  # rows, cols
    assert ac == br
    c = torch.zeros(ar, bc)

    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                c[i, j] += a[i, k] * b[k, j]
    return c


m1 = X_test[:5]
m2 = weights
print(m1.shape, m2.shape)
matmul_slow(m1, m2)


# =============================================================================
# Matrix multiplication [Element-Wise][Fast]
# =============================================================================
def matmul_fast(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            c[i, j] = (a[i, :] * b[:, j]).sum()

    return c


matmul_fast(m1, m2)
torch.allclose(matmul_fast(m1, m2), matmul_slow(m1, m2), rtol=1e-3, atol=1e-5)


# =============================================================================
# Matrix multiplication [Bradcasting][Faster]
# =============================================================================
def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        # print(a[i, :].unsqueeze(-1))  # [i, :]  == [i] we can ommit ,:
        # a[i].unsqueeze(-1) transform 1 dim tensor to 2 dim tensor
        c[i] = (a[i].unsqueeze(-1) * b).sum(dim=0)
    return c


matmul(m1, m2)

# =============================================================================
# Matrix multiplication [PyTorch]
# =============================================================================
m1.matmul(m2)
