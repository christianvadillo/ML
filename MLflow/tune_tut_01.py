# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:34:34 2020

@author: 1052668570
"""
import torch
import torch.optim as optim
import ray
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp


def train_mnist(config):
    train_ld, test_ld = get_data_loaders()
    model = ConvNet()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'])

    for i in range(30):
        train(model, optimizer, train_ld, torch.device('cpu'))
        acc = test(model, test_ld, torch.device('cpu'))
        tune.report(mean_accuracy=acc)


space = {
    'lr': hp.loguniform('lr', 1e-10, 0.1),
    'momentum': hp.uniform('momentum', 0.1, 0.9)
    }

hp_search = HyperOptSearch(
    space, max_concurrent=2, metric='mean_accuracy', mode='max')

# ray.init(object_store_memory=78643200)
analysis = tune.run(
    train_mnist,
    num_samples=10,
    search_alg=hp_search)
    # config={'lr': tune.grid_search([0.001, 0.01, 0.1])})

print('Best config: ', analysis.get_best_config(metric='mean_accuracy'))

df = analysis.dataframe()