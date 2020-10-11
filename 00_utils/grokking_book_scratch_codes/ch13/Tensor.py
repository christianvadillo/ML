# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:27:20 2020

@author: 1052668570

Basic framework for deep learning 

"""

import numpy as np


class Tensor(object):
    """ creators: list containing any tensors used in the creation of the
                    current tensor.
        creation_op: feature that stores the instructions creators used in the
                    creation of the process """

    def __init__(self, data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id_=None):

        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.children = {}

        if id_ is None:
            self.id_ = np.random.randint(0, 100000)
        else:
            self.id_ = id_

        if creators is not None:
            for c in creators:
                # Track how many children a tensor has
                # print('-'*10)
                # print(f'id:{self.id_}, data:{self.data}')
                # print(f'creator {c}')
                # print(f'children {c.children}')
                if self.id_ not in c.children:
                    c.children[self.id_] = 1
                else:
                    c.children[self.id_] += 1
                # print(f'children {c.children}')
                # print('-'*10)

    def all_children_grads_accounted_for(self):
        """ Checks whether a tensor has received the correct number of
        gradients from each child """
        for id_, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        """ autograd implementation, propragate the gradients along all
        tensors """
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            if grad_origin is not None:
                # check if we can backpropragate
                if self.children[grad_origin.id_] == 0:
                    raise Exception('cannot backprop more than once')
                else:
                    self.children[grad_origin.id_] -= 1

            # Accumulates gradients from several children
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            # grads must not have grads of their own
            assert grad.autograd == False

            # only continue backpropping if there's something to
            # backprop into and if all gradients (from children)
            # are accounted for override waiting for children if
            # "backprop" was called on this variable directly
            if (self.creators is not None and
                (self.all_children_grads_accounted_for() or
                 grad_origin is None)):
                # Call backward to each Ten
                if self.creation_op == 'add':
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == 'sub':
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)

                if self.creation_op == 'mul':
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)

                if self.creation_op == 'mm':
                    activation = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    activation.backward(new)
                    new = self.grad.transpose().mm(activation).transpose()
                    weights.backward(new)

                if self.creation_op == 'tranpose':
                    self.creators[0].backward(self.grad.tranpose())

                if 'sum' in self.creation_op:
                    dim = int(self.creation_op.split('_')[1])
                    ds = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim, ds))

                if 'expand' in self.creation_op:
                    dim = int(self.creation_op.split('_')[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if self.creation_op == 'neg':
                    self.creators[0].backward(self.grad.__neg__())
                    
                if self.creation_op == 'sigmoid':
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))
                
                if self.creation_op == 'tanh':
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))
                    
                if self.creation_op == 'index_select':
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))
                
                if self.creation_op == 'cross_entropy':
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))

    def __add__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='add')
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='sub')
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='mul')
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op='sum_'+str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op='expand_'+str(dim))
        return Tensor(new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op='transpose')
        return Tensor(self.data.transpose())

    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op='mm')
        return Tensor(self.data.dot(x.data))

    # Nonlinearity Layers
    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op='sigmoid')
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op='tanh')
        return Tensor(np.tanh(self.data))

    def index_select(self, indices):
        """ Support for indexing """
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op='index_select')
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])
    
    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), - 1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()
        
        if self.autograd:
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op='cross_entropy')
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out
        return Tensor(loss)
    
    
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op='neg')
        return Tensor(self.data * -1)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

    @property
    def shape(self):
        return self.data.shape


# a = Tensor([1, 2, 3, 4, 5], autograd=True)
# b = Tensor([2, 2, 2, 2, 2], autograd=True)
# c = Tensor([5, 4, 3, 2, 1], autograd=True)

# d = a + b + c
# e = a + c
# f = d + e
# g = d + (-f)


# g.backward(Tensor(np.array([1, 1, 1, 1, 1])))
# # f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
# print(b.grad.data == np.array([2, 2, 2, 2, 2]))