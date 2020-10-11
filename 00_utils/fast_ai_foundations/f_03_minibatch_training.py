# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:08:33 2020

@author: 1052668570
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from util_functions import get_data

mpl.rcParams['image.cmap'] = 'gray'

X_train, y_train, X_test, y_test = get_data()

# Shape
n, m = X_train.shape
o = y_train.unique()
n_hidden = 50
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
# Torch Model
# =============================================================================
class Model(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(Model, self).__init__()

        self.layers = [nn.Linear(n_in, n_hidden),
                       nn.ReLU(),
                       nn.Linear(n_hidden, n_out)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


model = Model(m, n_hidden, 10)
pred = model(X_train)


# =============================================================================
#  Cross Entropy Loss
# =============================================================================
# Softmax
def log_softmax(x):
    """ In practice we will need the log of the softmax when we calculate
    the loss """
    top = x.exp()
    bottom = x.exp().sum(-1, keepdim=True)
    return (top/bottom).log()


soft_max_pred = log_softmax(pred)


# Negative likelihood"
def nll(inp, target):
    """ Negative likelihood"""
    return -inp[range(target.shape[0]), target].mean()


# Loss
loss = nll(soft_max_pred, y_train)
print(loss)

# =============================================================================
# PyTorch implementation
# =============================================================================
torch_loss = F.nll_loss(F.log_softmax(pred, -1), y_train)

torch.allclose(loss, torch_loss, rtol=1e-3, atol=1e-5)

# In PyTorch, log_softmax and nll_loss are combined in one optimized function,
# F.cross_entropy

cross_entropy = F.cross_entropy(pred, y_train)
torch.allclose(loss, cross_entropy, rtol=1e-3, atol=1e-5)

# =============================================================================
# =============================================================================
# Basic Training Loop
# =============================================================================
# =============================================================================
model = Model(m, n_hidden, 10)


def accuracy(out, targets):
    """ Accuracy metric"""
    return (torch.argmax(out, dim=1) == targets).float().mean()


loss_func = F.cross_entropy
batch_size = 64
lr = 0.5
epochs = 10
train_accuracies = []
train_losses = []

batches = ((n-1)//batch_size + 1)
for epoch in range(epochs):
    train_acc = []
    train_loss = []

    for i in range(batches):
        start = batch_size*i
        end = batch_size*(i+1)
        x_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward
        outputs = model(x_batch)
        loss = loss_func(outputs, y_batch)

        # Backward
        loss.backward()

        # Updates of the parameters with no grad
        # Because their are not part of the gradient
        # calculations
        with torch.no_grad():
            for layer in model.layers:
                if hasattr(layer, 'weight'):
                    layer.weight -= layer.weight.grad * lr
                    layer.bias -= layer.bias.grad * lr
                    # Since the backward() function accumulates gradients,
                    # and you don’t want to mix up gradients between
                    # minibatches, you have to zero them out at the start 
                    # of a new minibatch. This is exactly like how a general 
                    # (additive) accumulator variable is initialized to 0 in
                    # code.
                    layer.weight.grad.zero_()
                    layer.bias.grad.zero_()

        train_loss.append(loss.item())
        train_acc.append(accuracy(outputs, y_batch).item())

    train_loss = np.mean(train_loss)
    train_acc = np.mean(train_acc)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f'Epoch: {epoch}/{epochs}')
    print(f'Train loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}')


# =============================================================================
# =============================================================================
# Basic Training Loop [REFACTORING CODE]
# =============================================================================
# =============================================================================
""" Behind the scenes PyTorch overrides the __setattr__ function in nn.Module
so that the submodules you define are properly registered as parameters of the
model
"""

class DummyModule():
    def __init__(self, n_in, n_hidden, n_out):
        self._modules = {}
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def __setattr__(self, name, value):
        """ To check if the set in the __init__ method is a layer, if it then
        added to the dictionary _modules"""
        if not name.startswith("_"):
            self._modules[name] = value
        super().__setattr__(name, value)

    def __repr__(self):
        return f'{self._modules}'

    def parameters(self):
        for layer in self._modules.values():
            for param in layer.parameters():
                yield param


mdl = DummyModule(m, n_hidden, 10)
print(mdl)
next(mdl.parameters())


# =============================================================================
# Using PyTorch with the behaviour described above

class Model(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def __call__(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


model = Model(m, n_hidden, 10)
for name, layer in model.named_children():
    print(f'{name}, {layer}')


def fit():
    loss_func = F.cross_entropy
    batch_size = 64
    lr = 0.5
    epochs = 10
    train_accuracies = []
    train_losses = []

    batches = ((n-1)//batch_size + 1)
    for epoch in range(epochs):
        train_acc = []
        train_loss = []
    
        for i in range(batches):
            start = batch_size*i
            end = batch_size*(i+1)
            x_batch = X_train[start:end]
            y_batch = y_train[start:end]
    
            # Forward
            outputs = model(x_batch)
            loss = loss_func(outputs, y_batch)
    
            # Backward
            loss.backward()
    
            # Updates of the parameters with no grad
            # Because their are not part of the gradient
            # calculations
            with torch.no_grad():
                for param in model.parameters():
                    param -= param.grad * lr
                    # Since the backward() function accumulates gradients,
                    # and you don’t want to mix up gradients between
                    # minibatches, you have to zero them out at the start 
                    # of a new minibatch. This is exactly like how a general 
                    # (additive) accumulator variable is initialized to 0 in
                    # code.
                    model.zero_grad()
    
            train_loss.append(loss.item())
            train_acc.append(accuracy(outputs, y_batch).item())
    
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
    
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
    
        print(f'Epoch: {epoch}/{epochs}')
        print(f'Train loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}')


# =============================================================================
# =============================================================================
# Registering modules using nn.ModuleList
# =============================================================================
# =============================================================================
layers = [nn.Linear(m, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 10)]


class SequentialModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


model = SequentialModel(layers)
print(model)
fit()

# =============================================================================
# Same but using nn.Sequential
model = nn.Sequential(nn.Linear(m, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, 10))
print(model)
fit()


# =============================================================================
# =============================================================================
# Refactoring the Update paramaters
""" Let's replace our previous manually coded optimization step:

    with torch.no_grad():
        for param in model.parameters():
            param -= param.grad * lr
            model.zero_grad()

   and instead use just:
     opt.step()
     opt.zero_grad()
"""
# =============================================================================
# =============================================================================


class Optimizer():
    def __init__(self, params, lr=0.5):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= param.grad * self.lr

    def zero_grad(self):
        for param in self.params:
            param.grad.data.zero_()


model = nn.Sequential(nn.Linear(m, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, 10))
print(model)
optimizer = Optimizer(model.parameters(), lr=0.01)

loss_func = F.cross_entropy
batch_size = 64
epochs = 10
train_accuracies = []
train_losses = []

batches = ((n-1)//batch_size + 1)
for epoch in range(epochs):
    train_acc = []
    train_loss = []

    for i in range(batches):
        start = batch_size*i
        end = batch_size*(i+1)
        x_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward
        outputs = model(x_batch)
        loss = loss_func(outputs, y_batch)

        # Backward
        loss.backward()  # calculate the gradients
        optimizer.step()  # update the params
        optimizer.zero_grad()  # reset the gradients

        train_loss.append(loss.item())
        train_acc.append(accuracy(outputs, y_batch).item())

    train_loss = np.mean(train_loss)
    train_acc = np.mean(train_acc)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f'Epoch: {epoch}/{epochs}')
    print(f'Train loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}')

# =============================================================================
# Using PyTorch with the behaviour described above
from torch import optim


def get_model():
    model = nn.Sequential(nn.Linear(m, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, 10))
    return model, optim.SGD(model.parameters(), lr=lr)

model, optimizer = get_model()

for epoch in range(epochs):
    train_acc = []
    train_loss = []

    for i in range(batches):
        start = batch_size*i
        end = batch_size*(i+1)
        x_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward
        outputs = model(x_batch)
        loss = loss_func(outputs, y_batch)

        # Backward
        loss.backward()  # calculate the gradients
        optimizer.step()  # update the params
        optimizer.zero_grad()  # reset the gradients

        train_loss.append(loss.item())
        train_acc.append(accuracy(outputs, y_batch).item())

    train_loss = np.mean(train_loss)
    train_acc = np.mean(train_acc)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f'Epoch: {epoch}/{epochs}')
    print(f'Train loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}')
