# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:08:40 2020

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
from torch.utils.data import DataLoader, TensorDataset

mpl.rcParams['image.cmap'] = 'gray'


def get_model(lr=0.1):
    model = nn.Sequential(nn.Linear(m, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, 10))
    return model, optim.Adam(model.parameters(), lr=lr)


def accuracy(out, targets):
    """ Accuracy metric"""
    return (torch.argmax(out, dim=1) == targets).float().mean()


class FastTensorDataLoader:
    """
    https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
    
    
X_train, y_train, X_val, y_val = get_data()

# Shape
n, m = X_train.shape
o = y_train.unique()
n_hidden = 50
# print(X_train)
# print(X_train.shape)
# print(y_train)
# print(y_train.shape)
# print(X_val)
# print(X_val.shape)
# print(y_val)
# print(y_val.shape)
# print(y_train.min(), y_train.max())

# img = X_train[0]
# img.view(28, 28).type
# plt.imshow(img.view(28, 28))

# =============================================================================
# Setting up the Loading Data 
# =============================================================================
batch_size = 64
train_dl = FastTensorDataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
val_dl = FastTensorDataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)

# =============================================================================
# Main loop training with Validation set inlcuded
# We will calculate and print the validation loss at the end of each epoch.
""" (Note that we always call model.train() before training, and model.eval()
 before inference, because these are used by layers such as nn.BatchNorm2d
 and nn.Dropout to ensure appropriate behaviour for these different phases.)"""
# This approach is not valid when the size of the batches varies
# =============================================================================
model, optimizer = get_model(lr=0.1)
loss_func = F.cross_entropy
epochs = 5

train_loss, train_acc  = np.zeros(epochs), np.zeros(epochs)
val_loss, val_acc  = np.zeros(epochs), np.zeros(epochs)


for epoch in range(epochs):
    tot_train_loss, tot_train_acc = 0., 0.,
    # Handle batchnorm / dropout
    model.train()
    # print(f'Model Training: {model.training}')
    for x_batch, y_batch in train_dl:
        # Forward
        outputs = model(x_batch)
        loss = loss_func(outputs, y_batch)
        
        # Backward
        loss.backward()  # calculate the gradients
        optimizer.step()  # update the params
        optimizer.zero_grad()  # reset the gradients

        tot_train_loss += loss.item()
        tot_train_acc += accuracy(outputs, y_batch).item()
        
    model.eval()
    # print(f'Model Training: {model.training}')
    with torch.no_grad():
        tot_val_loss, tot_val_acc = 0., 0.,
        for x_batch, y_batch in val_dl:
            outputs = model(x_batch)
            tot_val_loss += loss_func(outputs, y_batch)
            tot_val_acc += accuracy(outputs, y_batch)
    
    n_train = len(train_dl)
    n_val = len(val_dl)
    
    tr_loss = tot_train_loss/n_train
    tr_acc = tot_train_acc/n_train
    v_loss = tot_val_loss/n_val
    v_acc = tot_val_acc/n_val
    
    train_loss[epoch] = tr_loss
    train_acc[epoch] = tr_acc
    
    val_loss[epoch] = v_loss
    val_acc[epoch] = v_acc

    print(f'Epoch: {epoch+1}/{epochs}')
    print(f'train_loss = {tr_loss:.4f}, train_accuracy = {tr_acc:.4f}')
    print(f'val_loss = {v_loss:.4f}, val_accuracy = {v_acc:.4f}')
    
plt.plot(train_loss)
plt.plot(val_loss)