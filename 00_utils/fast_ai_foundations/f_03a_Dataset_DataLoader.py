# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:53:20 2020

@author: 1052668570
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
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


"""
Dataset
It's clunky to iterate through minibatches of x and y values separately:

x_batch = x_train[start:end]
y_batch = y_train[start:end]
Instead, let's do these two steps together, by introducing a Dataset class:

x_batch,y_batch = train_ds[i*batch_size : (i+1)*batch_size]

"""


class Dataset():
    """ Pure Python """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


train_ds = Dataset(X_train, y_train)
test_ds = Dataset(X_test, y_test)
x_train_batch, y_train_batch = train_ds[0]
x_test_batch, y_test_batch = test_ds[0]

assert len(train_ds) == len(X_train)
assert len(test_ds) == len(X_test)


x_train_batch, y_train_batch = train_ds[0:5]
x_test_batch, y_test_batch = test_ds[0:5]
assert x_train_batch.shape == (5, 28*28)
assert x_test_batch.shape == (5, 28*28)
assert y_train_batch.shape == (5, )
assert y_test_batch.shape == (5, )


def get_model(lr=0.1):
    model = nn.Sequential(nn.Linear(m, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, 10))
    return model, optim.Adam(model.parameters(), lr=lr)


def accuracy(out, targets):
    """ Accuracy metric"""
    return (torch.argmax(out, dim=1) == targets).float().mean()


model, optimizer = get_model()

loss_func = F.cross_entropy
batch_size = 64
epochs = 5
train_accuracies = []
train_losses = []

batches = ((n-1)//batch_size + 1)
for epoch in range(epochs):
    train_acc = []
    train_loss = []

    for i in range(batches):
        start = batch_size*i
        end = batch_size*(i+1)
        x_batch, y_batch = train_ds[start:end]

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
# =============================================================================
# Refactoring Batch loop code using DataLoader class
# =============================================================================
# =============================================================================

class DataLoader():
    """ Pure Python """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[i:i+self.batch_size]


train_dl = DataLoader(train_ds, batch_size)
test_dl = DataLoader(test_ds, batch_size)

x_batch, y_batch = next(iter(train_dl))
assert x_batch.shape == (batch_size, 28*28)

# plt.imshow(x_batch[0].view(28, 28))
# plt.title(y_batch[0].item())


def fit():
    train_accuracies = []
    train_losses = []
    for epoch in range(epochs):
        train_acc = []
        train_loss = []

        for x_batch, y_batch in train_dl:
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


model, optimizer = get_model(lr=0.01)
fit()


# =============================================================================
# =============================================================================
# Adding Random Sampling to the batches
# =============================================================================
""" We want our training set to be in a random order, and that order should
 differ each iteration. But the validation set shouldn't be randomized."""
# =============================================================================


class Sampler():
    def __init__(self, dataset, batch_size, shuffle=False):
        self.n = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.idxs = torch.randperm(self.n)\
            if self.shuffle else torch.arange(self.n)

        for i in range(0, self.n, self.batch_size):
            yield self.idxs[i:i+self.batch_size]


small_dataset = Dataset(*train_ds[:10])
s = Sampler(small_dataset, 3, False)
[item for item in s]
s = Sampler(small_dataset, 3, True)
[item for item in s]


# We can replace DataLoader with a sampler
def collate(b):
    xs, ys = zip(*b)
    return torch.stack(xs), torch.stack(ys)


class DataLoader():
    def __init__(self, dataset, sampler, collate_fn=collate):
        self.dataset = dataset
        self.sampler = sampler
        self.collate_fn = collate_fn

    def __iter__(self):
        for sample in self.sampler:
            yield self.collate_fn([self.dataset[i] for i in sample])


train_ds = Dataset(X_train, y_train)
test_ds = Dataset(X_test, y_test)

# Suffling?
train_samp = Sampler(train_ds, batch_size, shuffle=True)
test_samp = Sampler(test_ds, batch_size, shuffle=False)

# Batch Iter
train_dl = DataLoader(train_ds, sampler=train_samp, collate_fn=collate)
test_dl = DataLoader(test_ds, sampler=test_samp, collate_fn=collate)

# xb, yb = next(iter(test_dl))
# plt.imshow(xb[0].view(28,28))
# yb[0]

model, optimizer = get_model(lr=0.1)
fit()

# =============================================================================
# PyTorch DataLoader
# =============================================================================
from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)

loss_func = F.cross_entropy
model, optimizer = get_model(lr=0.01)


train_accuracies = []
train_losses = []
for epoch in range(epochs):
    train_acc = []
    train_loss = []
    # x_batch, y_batch = next(iter(train_dl))
    for x_batch, y_batch in train_dl:
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

