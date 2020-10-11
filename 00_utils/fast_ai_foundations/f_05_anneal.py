# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:08:45 2020

@author: 1052668570
"""

import math
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
from f_04_callbacks import Runner, AvgStatsCallback, Callback, listify
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


class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model = model
        self.optimizer = opt
        self.loss_func = loss_func
        self.data = data




# =============================================================================
# Getting the data
# =============================================================================
X_train, y_train, X_val, y_val = get_data()
# Shape
n, m = X_train.shape
k = len(y_train.unique())
n_hidden = 50


# =============================================================================
# Setting up the Generators
# =============================================================================
batch_size = 64

train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      drop_last=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

# Store data generators
data = DataBunch(train_dl, val_dl, k)

loss_func = F.cross_entropy
model, optimizer = get_model(lr=0.1)


# =============================================================================
# Creating the learner to fit
# =============================================================================
def create_learner(model, optimizer, loss_func, data):
    return Learner(model, optimizer, loss_func, data)


learn = create_learner(model, optimizer, loss_func, data)

# =============================================================================
# Creating the Runner
# =============================================================================
run = Runner([AvgStatsCallback([accuracy])])
run.fit(3, learn)

# =============================================================================
# 
# =============================================================================
from functools import partial
acc_cbf = partial(AvgStatsCallback, accuracy)
run = Runner(callback_funcs=acc_cbf)
run.fit(1, learn)

run.avg_stats.valid_stats.avg_stats


# =============================================================================
# Annealing
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/05_anneal.ipynb
""" We define two new callbacks: the Recorder to save track of the loss and
 our scheduled learning rate, and a ParamScheduler that can schedule any
 hyperparameter as long as it's registered in the state_dict of the optimizer.
 """
# =============================================================================


class Recorder(Callback):
    def begin_fit(self):
        self.learning_rates = []
        self.losses = []

    def after_batch(self):
        if not self.in_train: return
        self.learning_rates.append(self.opt.param_groups[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self): 
        plt.plot(self.learning_rates)

    def plot_loss(self):
        plt.plot(self.losses)


class ParamScheduler(Callback):
    """ Hyperparameters scheduling"""
    _order = 1

    def __init__(self, pname, sched_func):
        self.pname = pname
        self.sched_func = sched_func

    def set_param(self):
        for pg in self.opt.param_groups:
            pg[self.pname] = self.sched_func(self.n_epochs/self.epochs)

    def begin_batch(self):
        if self.in_train:
            self.set_param()


# =============================================================================
# 
""" 
Schedule learning rate:
    * Linear
    * cos
    * no
    * exp
 """
# =============================================================================

# decorator
def annealer(f):
    def _inner(start, end):
        return partial(f, start, end)
    return _inner

@annealer
def sched_line(start, end, pos):
    "Simple linear scheduling "
    return start + pos*(end-start)

@annealer
def sched_no(start, end, pos):
    return start

@annealer
def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

@annealer
def sched_exp(start, end, pos):
    return start * (end/start) ** pos


def cos_1cycle_annel(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]


#This monkey-patch is there to be able to plot tensors
torch.Tensor.ndim = property(lambda x: len(x.shape))


annealings = "NO LINEAR COS EXP".split()

a = torch.arange(0, 100)
p = torch.linspace(0.01, 1, 100)

fns = [sched_no, sched_line, sched_cos, sched_exp]
for fn, t in zip(fns, annealings):
    f = fn(2, 1e-2)
    plt.plot(a, [f(o) for o in p], label=t)
plt.legend();


""" In practice, we'll often want to combine different schedulers, 
the following function does that: it uses scheds[i]
 for pcts[i] of the training. """


def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        if idx == 2: idx = 1
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner

""" Here is an example: use 30% of the budget to go from 0.3 to 0.6
 following a cosine, then the last 70% of the budget to go from 0.6 to 0.2,
 still following a cosine. """
 
sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)])
plt.plot(a, [sched(o) for o in p])


cbfs = [Recorder,
        partial(AvgStatsCallback, accuracy),
        partial(ParamScheduler, 'lr', sched)]

model, optimizer = get_model(lr=0.1)
learn = create_learner(model, optimizer, loss_func, data)
run = Runner(callback_funcs=cbfs)
run.fit(3, learn)

# Plots
run.recorder.plot_lr()
run.recorder.plot_loss()
