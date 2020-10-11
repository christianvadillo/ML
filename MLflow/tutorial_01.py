# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:50:38 2020

@author: 1052668570
"""

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(pred, targ):
    rmse = np.sqrt(mean_absolute_error(targ, pred))
    mae = mean_absolute_error(targ, pred)
    r2 = r2_score(targ, pred)
    return rmse, mae, r2


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(40)

    csv_url =\
        'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    try:
        data = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception(
            "Unable to download training/test csv. Error %s", e)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    
    # The predicted column is "quality" which is a scalar from [3, 9]
    x_train = train.drop(["quality"], axis=1)
    x_test = test.drop(["quality"], axis=1)
    
    y_train = train[["quality"]]
    y_test = test[["quality"]]
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(x_train, y_train)
        
        preds = lr.predict(x_test)
        
        (rmse, mae, r2) = eval_metrics(preds, y_test)
        
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        
        mlflow.log_param('alpha', alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
    

