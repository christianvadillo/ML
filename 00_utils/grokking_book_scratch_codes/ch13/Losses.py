# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:45:56 2020

@author: 1052668570
"""
from Layers import Layer


class CrossEntropyLoss:
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.cross_entropy(target)


class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)