# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:48:34 2020

@author: 1052668570
"""
import numpy as np

from Tensor import Tensor
from Layers import Sequential, Linear, Embedding, Tanh, Sigmoid
from Optimizers import SGD
from Losses import MSELoss, CrossEntropyLoss



# =============================================================================
# train ann using autograd
# =============================================================================

np.random.seed(0)

data = Tensor(np.array([1, 2, 1, 2]), autograd=True)

target = Tensor(np.array([
    [0],
    [1],
    [0],
    [1]
    ]), autograd=True)

# w = list()
# w.append(Tensor(np.random.rand(2, 3), autograd=True))
# w.append(Tensor(np.random.rand(3, 1), autograd=True))

# Embedding
# Sequential model
model = Sequential([Embedding(3, 3),
                    Tanh(),
                    Linear(3, 4)
                    ])


# Optimizer
optim = SGD(parameters=model.get_parameters(), alpha=0.1)
# Loss function
criterion = CrossEntropyLoss()

for i in range(10):
    # pred = data.mm(w[0]).mm(w[1])
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)
