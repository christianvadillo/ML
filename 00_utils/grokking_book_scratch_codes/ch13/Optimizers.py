# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:46:04 2020

@author: 1052668570
"""


class SGD:
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zeros(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha

            if zero:
                p.grad.data *= 0
