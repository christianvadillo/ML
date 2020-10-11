# -*- coding: utf-8 -*-

import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sns.set_style("whitegrid")

data = pd.read_csv("../data/classification/kyphosis.csv")

data.head()
data.info()

sns.pairplot(data, hue='Kyphosis', diag_kind='kde')

sns.countplot(data['Kyphosis'])
# =============================================================================
# Feature engineering
# =============================================================================
# map_dict = {'absent': 0, 'present': 1}
# data['Kyphosis'] = data['Kyphosis'].map(map_dict)

# =============================================================================
# Splitting data
# =============================================================================

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# =============================================================================
# DT Model
# =============================================================================
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# =============================================================================
# Evaluating model
# =============================================================================
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# =============================================================================
# RF Model
# =============================================================================
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)

# =============================================================================
# Evaluating model
# =============================================================================
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# =============================================================================
# Visualizing
# =============================================================================
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
import os

features = list(data.columns[1:])
export_graphviz(model, out_file = 'tree.dot', 
                feature_names = features, 
                rounded = True, 
                precision = 2,
                # class_names   = ["Normal", "Falla"], 
                filled = True,)

(graph, ) = pydot.graph_from_dot_file('tree.dot')
os.system(f'dot -Tpng tree.dot -o tree_test.png')


# GENERATING DECISION TREES from RF
i_tree = 0
for tree_in_forest in rf.estimators_:
    export_graphviz(tree_in_forest, out_file = 'tree.dot', 
                    feature_names = features, 
                    rounded = True, 
                    precision = 2,
                    # class_names   = ["Normal", "Falla"], 
                    filled = True,)

    (graph, ) = pydot.graph_from_dot_file('tree.dot')
#    name = 'tree' + str(i_tree)
#    graph.write_png(name+  '.png')
    os.system(f'dot -Tpng tree.dot -o tree_{i_tree}.png')
    i_tree += 1