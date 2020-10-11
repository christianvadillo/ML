# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:19:51 2020

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
k = len(y_train.unique())
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

train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      drop_last=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

# train_dl = FastTensorDataLoader(X_train, y_train, batch_size=batch_size, shuffle=False)
# val_dl = FastTensorDataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)

# =============================================================================
# Main loop training with Validation set inlcuded
# We will calculate and print the validation loss at the end of each epoch.
""" (Note that we always call model.train() before training, and model.eval()
 before inference, because these are used by layers such as nn.BatchNorm2d
 and nn.Dropout to ensure appropriate behaviour for these different phases.)"""
# This approach is not valid when the size of the batches varies
# =============================================================================
model, optimizer = get_model(lr=0.01)
loss_func = F.cross_entropy
epochs = 100

train_loss, train_acc = np.zeros(epochs), np.zeros(epochs)
val_loss, val_acc = np.zeros(epochs), np.zeros(epochs)


def train(train_dl, val_dl, model, epochs, optimizer, loss_func):
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


# train(train_dl, val_dl, model, 5, optimizer, loss_func)

# plt.plot(train_loss)
# plt.plot(val_loss)


# =============================================================================
# =============================================================================
# Let's Refactor the train to something that looks like this:
# fit(1, learn)
# https://medium.com/@lankinen/fast-ai-lesson-9-notes-part-2-v3-ca046a1a62ef
# https://course.fast.ai/videos/?lesson=9
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/04_callbacks.ipynb
# =============================================================================
# =============================================================================

# Start creating a new class for store our DataLoader
class DataBunch:
    def __init__(self, train_dl, valid_dl, n_out=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.n_out = n_out

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset


data = DataBunch(train_dl, val_dl, k)


# Then refactor the function for the model creation with a new class (Learner)
class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model = model
        self.optimizer = opt
        self.loss_func = loss_func
        self.data = data


def get_model(data, lr=0.5, n_hidden=50):
    m = data.train_ds.tensors[0].shape[1]
    model = nn.Sequential(nn.Linear(m, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, data.n_out)
                          )
    optimizer = optim.SGD(model.parameters(), lr=lr)
    return model, optimizer


learn = Learner(*get_model(data, lr=0.5, n_hidden=50), loss_func, data)


# Finally refactor the train function with the new objects
def train(epochs, learn):
    for epoch in range(epochs):
        tot_train_loss, tot_train_acc = 0., 0.,
        # Handle batchnorm / dropout
        learn.model.train()
        # print(f'Model Training: {model.training}')
        for x_batch, y_batch in learn.data.train_dl:
            # Forward
            outputs = learn.model(x_batch)
            loss = learn.loss_func(outputs, y_batch)

            # Backward
            loss.backward()  # calculate the gradients
            learn.optimizer.step()  # update the params
            learn.optimizer.zero_grad()  # reset the gradients

            tot_train_loss += loss.item()
            tot_train_acc += accuracy(outputs, y_batch).item()

        learn.model.eval()
        # print(f'Model Training: {model.training}')
        with torch.no_grad():
            tot_val_loss, tot_val_acc = 0., 0.,
            for x_batch, y_batch in learn.data.valid_dl:
                outputs = learn.model(x_batch)
                tot_val_loss += learn.loss_func(outputs, y_batch)
                tot_val_acc += accuracy(outputs, y_batch)

        n_train = len(learn.data.train_dl)
        n_val = len(learn.data.valid_dl)

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


# train(5, learn)


# =============================================================================
# =============================================================================
# Let's add callbacks using Callback() and CallbackHandler()
# =============================================================================
# =============================================================================
import re

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])(A-Z)')


def camel2snake(name):
    """ Change name from camel to snake style.
    camel2snake('DeviceDataLoader') --> 'device_data_loader'
    """
    s1 = re.sub(_camel_re1, r'\1_\2', name)

    return re.sub(_camel_re2, 'r\1_\2', s1).lower()


class Callback():
    _order = 0
    
    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')


class TrainEvalCallback(Callback):
    """ This first callback is reponsible to switch the model
    back and forth in training or validation mode, as well as
    maintaining a count of the iterations, or the percentage
    of iterations ellapsed in the epoch."""

    def begin_fit(self):
        self.run.n_epochs = 0.
        self.run.n_iter = 0

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter += 1

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


class TestCallback(Callback):
    _order = 1
    def after_step(self):
        if self.train_eval.n_iters >= 10:
            return True

        
TrainEvalCallback().name
TestCallback().name

# =============================================================================
# RUNNER CLASS
# =============================================================================
from typing import *


def listify(obj):
    if obj is None: return []
    if isinstance(obj, list): return obj
    if isinstance(obj, str): return [obj]
    if isinstance(obj, Iterable): return list(obj)
    return [obj]


class Runner():
    def __init__(self, callbacks=None, callback_funcs=None):
        callbacks = listify(callbacks)
        for cbf in listify(callback_funcs):
            callback = cbf()
            setattr(self, callback.name, callback)
            callbacks.append(callback)
        self.stop = False
        self.callbacks = [TrainEvalCallback()] + callbacks
        
    @property
    def opt(self):
        return self.learn.optimizer
    
    @property
    def model(self):
        return self.learn.model
    
    @property
    def loss_func(self):
        return self.learn.loss_func
    
    @property
    def data(self):
        return self.learn.data
    
    def one_batch(self, x_batch, y_batch):
        self.x_batch = x_batch
        self.y_batch = y_batch
        if self('begin_batch'): return
        self.pred = self.model(self.x_batch)
        if self('after_pred'): return
        self.loss = self.loss_func(self.pred, self.y_batch)
        if self('after_loss') or not self.in_train: return
        self.loss.backward()
        if self('after_backward'): return
        self.opt.step()
        if self('after_step'): return
        self.opt.zero_grad()
        
    def all_batches(self, dl):
        self.iters = len(dl)
        for x_batch, y_batch in dl:
            if self.stop: break
            self.one_batch(x_batch, y_batch)
            self('after_batch')
        self.stop=False   
    
    def fit(self, epochs, learn):
        self.epochs = epochs
        self.learn = learn

        try:
            for callback in self.callbacks: callback.set_runner(self)
            if self('begin_fit'): return
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                with torch.no_grad(): 
                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                if self('after_epoch'): break
            
        finally:
            self('after_fit')
            self.learn = None

    def __call__(self, cb_name):
        for cb in sorted(self.callbacks, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f(): return True
        return False
    
    
class AvgStats():
    """ Third callback: how to compute metrics. """
    def __init__(self, metrics, in_train): 
        self.metrics = listify(metrics)
        self.in_train = in_train
    
    def reset(self):
        self.tot_loss = 0.
        self.count = 0
        self.tot_mets = [0.] * len(self.metrics)
        
    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.x_batch.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.y_batch) * bn


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
        
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)
  

# learn = Learner(*get_model(data, lr=0.5, n_hidden=50), loss_func, data)
# stats = AvgStatsCallback([accuracy])
# run = Runner(callbacks=stats)

# run.fit(2, learn)
