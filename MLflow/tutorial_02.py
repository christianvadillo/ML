# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:29:32 2020

@author: 1052668570
"""
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
# from mlflow.tracking import MlflowClient

import sklearn.datasets as datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# TRACKING_URI = 'http://127.0.0.1/'
# mlflow.set_tracking_uri(TRACKING_URI)
# client = mlflow.tracking.MlflowClient(TRACKING_URI)

#loading the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


def send_to_mlflow(exp_id, run_name, model, x, y):
    run = mlflow.start_run(experiment_id=exp_id, run_name=run_name)
    
    for key, val in model.get_params().items():
        print(key, val)
        mlflow.log_param(key, val)
    
    pred = model.predict(x)
    accuracy = accuracy_score(y, pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y, pred)
   
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision[1])
    mlflow.log_metric('recall', recall[1])
    mlflow.log_metric('fscore', fscore[1])
    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(model, 'models/' + run_name + '/')
    else:
         mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()


def fit_model(model, x, y):
    model.fit(x, y)
    return model


log = LogisticRegression()
model = fit_model(log, train_x, train_y)
send_to_mlflow(0, 'logistic_base', log, test_x, test_y)

nb = MultinomialNB()
model = fit_model(nb, train_x, train_y)
send_to_mlflow(0, 'naive_bayes_base', nb, test_x, test_y)

rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=2)
model = fit_model(rf, train_x, train_y)
send_to_mlflow(0, 'rf_base', rf, test_x, test_y)