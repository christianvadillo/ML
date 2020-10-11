# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:56:50 2020

@author: 1052668570
"""
import torch
import math
from util_functions import get_data, normalize

X_train, y_train, X_test, y_test = get_data()

# =============================================================================
# Normalize
# =============================================================================
train_mean, train_std = X_train.mean(), X_train.std()
print(train_mean, train_std)
X_train = normalize(X_train, train_mean, train_std)
X_test = normalize(X_test, train_mean, train_std)
print(X_train.mean(), X_train.std())


def test_near_zero(a, tol=1e-3):
    assert a.abs()<tol, f'Near Zero: {a}'


print(test_near_zero(X_train.mean()))
print(test_near_zero(1 - X_train.std()))

# =============================================================================
# Shapes 
# =============================================================================
n, m = X_train.shape
c = y_train.max()+1
print(n, m, c)

# =============================================================================
# Fully Connected [Basic Architecture]
# =============================================================================
n_hidden = 50

# Parameters
# he_init/ kaiming init  # Read 2.2 of ResNet paper
w1 = torch.randn(m, n_hidden) * math.sqrt(2/m)
b1 = torch.zeros(n_hidden)
w2 = torch.randn(n_hidden, 1) * math.sqrt(2/m)
b2 = torch.zeros(1)


def lin(x, w, b):
    #  @ <-- perform matrix multiplication
    return x@w + b


def relu(x):
    # Change negatives to 0.
    return x.clamp_min_(0.)


t = relu(lin(X_train, w1, b1))
print(t.mean(), t.std())

# =============================================================================
# Using PyTorch to initialize weights
# =============================================================================
from torch.nn import init

w1 = torch.zeros(m, n_hidden)
# 'fan_out', to preserve variances on the backward phase
init.kaiming_normal_(w1, mode='fan_out')
t = relu(lin(X_train, w1, b1))
print(t.mean(), t.std())


# =============================================================================
# Creating Linear with PyTorch
# =============================================================================
import torch.nn
torch.nn.Linear(m, n_hidden).weight.shape  # Weight shape


# =============================================================================
# Alternative Relu
# =============================================================================
def relu(x):
    return x.clamp_min(0.) - 0.5


w1 = torch.randn(m, n_hidden) * math.sqrt(2. / m)
t1 = relu(lin(X_test, w1, b1))
# Reduce mean and increase std [Which is deseable]
print(t1.mean(), t1.std())


# =============================================================================
# Forward Pass
# =============================================================================
def forward(x):
    l1 = lin(x, w1, b1)
    print(l1.shape)
    l2 = relu(l1)
    print(l2.shape)
    l3 = lin(l2, w2, b2)
    return l3


forward(X_test)

# =============================================================================
# Loss Function: MSE [Provisional]
# =============================================================================
forward(X_test).shape  # torch.Size([10000, 1])
# We need to put outputs in proper shape in order to use MSE


def mse(outputs, targets):
    print(outputs.squeeze(-1).shape)
    print(targets.shape)
    return (outputs.squeeze(-1) - targets).pow(2).mean()


pred = forward(X_test)

mse(pred, y_test)

# =============================================================================
# Gradients and Backward pass
# =============================================================================
# Start with the very last function, in this case the loss function
# y_hat = mse(lin2(relu(lin1(x))), y)


def mse_grad(inp, targ):
    # inp: previous layer
    # targ: predicts
    # grad of Loss with respect to output of previous layer
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]


def relu_grad(inp, out):
    # inp: previous layer
    # out: next layer
    # grad of relu with respect to input activations
    inp.g = (inp > 0).float() * out.g


def lin_grad(inp, out, w, b):
    # inp: previous layer
    # out: next layer
    # grad of outputs with respect to inputs
    inp.g = out.g @ w.t()
    # grad of outputs with respect to weights
    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)
    # grad of outputs with respect to biases
    b.g = out.g.sum(0)


def forward_and_backward(inp, targ):
    # Forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2

    loss = mse(out, targ)

    # Backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)


forward_and_backward(X_train, y_train)

# Save for testing again later
w1g = w1.g.clone()
w2g = w2.g.clone()
b1g = b1.g.clone()
b2g = b2.g.clone()
ig = X_train.g.clone()


# =============================================================================
# Use PyTorch and autogrado to compare results
# =============================================================================
# Clone weights, bias and output
xt2 = X_train.clone().requires_grad_(True)
w12 = w1.clone().requires_grad_(True)
w22 = w2.clone().requires_grad_(True)
b12 = b1.clone().requires_grad_(True)
b22 = b2.clone().requires_grad_(True)


def forward_torch(inp, targ):
    l1 = inp @ w12 + b12
    l2 = relu(l1)
    out = l2 @ w22 + b22

    return mse(out, targ)


loss = forward_torch(xt2, y_train)
loss.backward()

print(
      torch.allclose(w22.grad, w2g, rtol=1e-3, atol=1e-5),
      torch.allclose(b22.grad, b2g, rtol=1e-3, atol=1e-5),
      torch.allclose(w12.grad, w1g, rtol=1e-3, atol=1e-5),
      torch.allclose(b12.grad, b1g, rtol=1e-3, atol=1e-5),
      torch.allclose(xt2.grad, ig, rtol=1e-3, atol=1e-5),
      )


# =============================================================================
# =============================================================================
# =============================================================================
# Layers as Class
# =============================================================================
class Relu:
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.) - 0.5
        return self.out

    def backward(self):
        self.inp.g = (self.inp > 0).float() * self.out.g


class Lin:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, inp):
        self.inp = inp
        self.out = self.inp@self.w + self.b  # logits
        return self.out

    def backward(self):
        # grad of outputs with respect to inputs
        self.inp.g = self.out.g @ self.w.t()
        # grad of outputs with respect to weights
        # self.w.g = (self.inp.unsqueeze(-1) * self.out.g.unsqueeze(1)).sum(0)
        self.w.g = self.inp.t() @ self.out.g  # Faster than ^
        # grad of outputs with respect to biases
        self.b.g = self.out.g.sum(0)


class Mse():
    def __call__(self, inp, targ):
        self.inp = inp
        self.targ = targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out

    def backward(self):
        self.inp.g = 2 * (self.inp.squeeze() - self.targ).\
                            unsqueeze(-1) / self.targ.shape[0]


class Model:
    def __init__(self, w1, b1, w2, b2):
        self.layers = ([Lin(w1, b1),
                        Relu(),
                        Lin(w2, b2)])
        self.loss = Mse()

    def __call__(self, x, targ):
        for layer in self.layers:
            x = layer(x)
        return self.loss(x, targ)

    def backward(self):
        self.loss.backward()
        for layer in reversed(self.layers):
            layer.backward()


w1.g, b1.g, w2.g, b2.g = [None]*4
model = Model(w1, b1, w2, b2)
loss = model(X_train, y_train)
model.backward()

print(
      torch.allclose(w2.g, w2g, rtol=1e-3, atol=1e-5),
      torch.allclose(b2.g, b2g, rtol=1e-3, atol=1e-5),
      torch.allclose(w1.g, w1g, rtol=1e-3, atol=1e-5),
      torch.allclose(b1.g, b1g, rtol=1e-3, atol=1e-5),
      )


# =============================================================================
# Layers as Class Using PyTorch
# =============================================================================
from torch import nn


class Model(nn.Module):
    def __init__(self, n_in, n_h, n_out):
        super(Model, self).__init__()
        self.layers = [nn.Linear(n_in, n_h),
                       nn.ReLU(),
                       nn.Linear(n_h, n_out)]
        self.loss = mse

    def __call__(self, x, targ):
        for layer in self.layers:
            x = layer(x)
        return self.loss(x.squeeze(), targ)


model = Model(m, n_hidden, 1)
loss = model(X_train, y_train)
loss.backward()