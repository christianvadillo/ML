# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:37:14 2020

@author: 1052668570
"""

import torch

x = torch.randn(512)
a = torch.randn(512, 512)

print(x.shape)
print(a.shape)

for _ in range(100):
    x = a@x

""" The problem you'll get with that is activation explosion: very soon,
your activations will go to nan. We can even ask the loop to break when
that first happens: """
print(x.mean(), x.std())

x = torch.randn(512)
a = torch.randn(512, 512)

for i in range(100):
    x = a@x
    if x.std() != x.std():
        break
print(i, x.mean(), x.std())
        