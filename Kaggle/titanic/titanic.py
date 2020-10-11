# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:43:02 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('grayscale')
mpl.rcParams['figure.figsize'] = 16, 8
mpl.rcParams['lines.linewidth'] = 0.7
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.edgecolor'] = 'grey'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'

pd.set_option('display.max_columns', None) # shows all columns
warnings.filterwarnings('ignore')

# !kaggle competitions download -c titanic 

# =============================================================================
# Load data
# =============================================================================
train_src = pd.read_csv("train.csv")
test_src = pd.read_csv("test.csv")
train_data = train_src.copy()
test_data = test_src.copy()
datasets = [train_data, test_data]

# =============================================================================
# Feature Engineering
# =============================================================================

for df in datasets:
    df['title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0])
    df['title'] = df['title'].replace(['Dona', 'Lady', 'the Countess',
                                       'Countess', 'Capt', 'Col', 'Don',
                                       'Dr', 'Major', 'Rev', 'Sir',
                                       'Jonkheer', 'Ms', 'Mme', 'Mlle'], 'Other',
                                      regex=True)


def family(x):
    if x < 2:
        return 'Single'
    elif x == 2:
        return 'Couple'
    elif x <= 4:
        return 'InterM'
    else:
        return 'Large'


for df in datasets:
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['family_size'] = df['family_size'].apply(family)


def sex_age(s, a):
    if (s == 'female') & (a < 18):
        return 'girl'
    elif (s == 'male') & (a < 18):
        return 'boy'
    elif (s == 'female') & (a > 18):
        return 'adult female'
    elif (s == 'male') & (a > 18):
        return 'adult male'
    elif (s == 'female') & (a > 50):
        return 'old female'
    else:
        return 'old male'


for df in datasets:
    df['Age'].fillna(df['Age'].median(), inplace=True)

test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

for df in datasets:
    df['sex_age'] = df.apply(lambda x: sex_age(x['Sex'], x['Age']), axis=1)

# Missing values [Cabin]
for df in datasets:
    df['Cabin'] = df['Cabin'].fillna(0)
    df['Cabin'] = df['Cabin'].fillna(0)
    df['hasCabin'] = df['Cabin'].apply(lambda x: 0 if x == 0 else 1)
    df['hasCabin'] = df['Cabin'].apply(lambda x: 0 if x == 0 else 1)

train_data.dropna(inplace=True)
PassengerId_test = test_data['PassengerId']

# Dropping useless columns [for now]
for df in datasets:
    df.drop(['PassengerId', 'Name', 'Ticket',
             'Cabin', 'Parch'], axis=1, inplace=True)

# =============================================================================
# Dummie variable
# =============================================================================
train_data = pd.get_dummies(data=train_data, columns=['title',
                                                      'Sex',
                                                      'Embarked',
                                                      'SibSp',
                                                      'hasCabin'],
                            drop_first=True)
test_data = pd.get_dummies(data=test_data, columns=['title',
                                                     'Sex',
                                                     'Embarked',
                                                     'SibSp',
                                                     'hasCabin'],
                           drop_first=True)

from sklearn.preprocessing import LabelEncoder
# Label Encoding family_size, Pclass
enc = LabelEncoder()
train_data['sex_age'] = enc.fit_transform(train_data['sex_age'])
train_data['family_size'] = enc.fit_transform(train_data['family_size'])
train_data['Pclass'] = enc.fit_transform(train_data['Pclass'])

test_data['sex_age'] = enc.fit_transform(test_data['sex_age'])    
test_data['family_size'] = enc.fit_transform(test_data['family_size'])
test_data['Pclass'] = enc.fit_transform(test_data['Pclass'])    


# =============================================================================
# PyTorch ANN
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def assert_close(a, b):
    # return torch.allclose(a, b, rtol=1e-03, atol=1e-05)
    return np.allclose(a, b, rtol=1e-03, atol=1e-05)


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
    
    
    
# Splitting the data
X = train_data.iloc[:, 1:].values
y = train_data.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.15, 
                                                    random_state=101)

# Scaling the data
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

print(assert_close(X_train_sc.mean(), 0))
print(assert_close(X_train_sc.std(), 1))

print(assert_close(X_test_sc.mean(), 0))
print(assert_close(X_test_sc.std(), 1))

# =============================================================================
# Setting up the generators
# =============================================================================
X_train_t = torch.from_numpy(X_train_sc.astype(np.float32))
X_test_t = torch.from_numpy(X_test_sc.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
y_test_t = torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))

# train_ds = TensorDataset(X_train_t, y_train_t)
# valid_ds = TensorDataset(X_test_t, y_test_t)

# train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
# valid_dl = DataLoader(train_ds, batch_size=16, shuffle=False)

train_dl = FastTensorDataLoader(X_train_t, y_train_t, batch_size=32, shuffle=True)
valid_dl = FastTensorDataLoader(X_test_t, y_test_t, batch_size=32, shuffle=False)
# =============================================================================
# Shapes
# =============================================================================
m = X_train_t.shape[1]
n_hidden = 32
k = 1
epochs=200
lr=0.001

# =============================================================================
# Setting up the model
# =============================================================================
class ModelANN(nn.Module):
    def __init__(self, n_in, n_h, n_out):
        self.n_out = n_out

        super(ModelANN, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, n_h)
        self.fc4 = nn.Linear(n_h, n_h)
        self.fc5 = nn.Linear(n_h, n_h)
        self.fcl = nn.Linear(n_h, n_out)

    def forward(self, x):
        for layer in model.children(): 
            if layer.out_features == self.n_out:
                return layer(x)
            else:
                x = F.relu(layer(x))
                x = F.dropout(x, p=0.6)


# =============================================================================
# Initialize weights
# =============================================================================
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        m.bias.data.fill_(0.01)


# =============================================================================
# Metric
# =============================================================================
def accuracy(outputs, targ):
    preds = outputs > 0.0
    return (preds.cpu() == targ.cpu()).numpy().sum()



# =============================================================================
# Setting up the device and moving model
# =============================================================================
model = ModelANN(n_in=m, n_h=n_hidden, n_out=k)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# =============================================================================
#  Loss function and Optimizer
# =============================================================================
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# =============================================================================
# Train loop
# =============================================================================
def train(model, criterion, optimizer, train_dl, valid_dl, epochs=10):
    train_losses, val_losses = np.zeros(epochs), np.zeros(epochs)
    train_accs, val_accs = np.zeros(epochs), np.zeros(epochs)
    best_acc = 0.
    
    for epoch in range(epochs):
        tot_train_loss = tot_train_acc = 0.
        tot_val_loss = tot_val_acc = 0.
        n_train = n_val = 0
        model.train()
        
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            # forward step
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            tot_train_loss += loss.item()
            
            # backward step
            loss.backward()
            optimizer.step()
            
            # accuracy
            tot_train_acc += accuracy(outputs, y_batch)
            n_train += y_batch.shape[0]
            
        # Validation Loss/Accuracy
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                tot_val_loss += criterion(outputs, y_batch)
                tot_val_acc += accuracy(outputs, y_batch)
                n_val += y_batch.shape[0]
    
        # Save epoch loss and accuracy
        train_loss = tot_train_loss / n_train
        val_loss = tot_val_loss / n_val
        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss
    
        # save epoch accuracy
        train_acc = tot_train_acc / n_train
        val_acc = tot_val_acc / n_val
        train_accs[epoch] = train_acc
        val_accs[epoch] = val_acc

        if val_acc > best_acc:
            best_acc = val_acc
            print(f'Saving model with accuracy: {best_acc}')
            torch.save(model, f'best-model-torch-{best_acc:.2f}.pt') 
            
        if (epoch+1)%20 == 0:
                print(f'Epoch: {epoch+1}/{epochs}')
                print(f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')
    
    return train_losses, val_losses, train_accs, val_accs


# Train
torch.backends.cudnn.benchmark = True
print(device)
train_losses, val_losses,\
     train_accs, val_accs = train(model, criterion, optimizer,
                                 train_dl, valid_dl, epochs=epochs)

plt.plot(train_losses, ls='--', label='train-loss')
plt.plot(val_losses, label='val-loss')
plt.legend()

from glob import glob
files = glob('best-model-torch-*')

best_models = [m for m in files if float(m.split('-')[3][0:4]) >= 0.85]


for m in best_models:
    best_torch_model = torch.load(m)

    with torch.no_grad():
        outputs = best_torch_model(X_test_t.to(device))
        y_pred = (outputs > 0).cpu().numpy()
    
    print('-'*20)
    print(m)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('-'*20)

# =============================================================================
# my_submission 
# =============================================================================
# test_data_sc = sc.transform(test_data)
# test_data_sc

# best_torch_model = torch.load('best-model-torch-0.89.pt')

# with torch.no_grad():
#     test_pred = best_torch_model(
#         torch.from_numpy(test_data_sc.astype(np.float32)).to(device))> 0.5
    
    
# test_pred = test_pred.cpu().numpy().astype(int).ravel()
# my_submission = pd.DataFrame({'PassengerId': PassengerId_test,
#                               'Survived': test_pred})

# my_submission.to_csv('./submission_t.csv', index=False, encoding='utf-8')
