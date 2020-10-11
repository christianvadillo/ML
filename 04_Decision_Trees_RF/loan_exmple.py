# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:03:00 2020

@author: 1052668570

For this project we will be exploring publicly available data from
LendingClub.com. Lending Club connects people who need money (borrowers)
with people who have money (investors). Hopefully, as an investor you would
want to invest in people who showed a profile of having a high probability
of paying you back. We will try to create a model that will help predict this.

Lending club had a very interesting year in 2016, so let's check out some of
their data and keep the context in mind. This data is from before they even
went public.

We will use lending data from 2007-2010 and be trying to classify and predict
whether or not the borrower paid back their loan in full. You can download the
data from here or just use the csv already provided.
It's recommended you use the csv provided as it has been cleaned of NA values.

Here are what the columns represent:

credit.policy:  1 if the customer meets the credit underwriting criteria
                of LendingClub.com, and 0 otherwise.
purpose: The purpose of the loan (takes values "credit_card",
         "debt_consolidation", "educational", "major_purchase",
         "small_business", and "all_other").
int.rate: The interest rate of the loan, as a proportion
            (a rate of 11% would be stored as 0.11). Borrowers judged by
            LendingClub.com to be more risky are assigned higher interest rates
installment: The monthly installments owed by the borrower 
                if the loan is funded.
log.annual.inc: The natural log of the self-reported annual
                income of the borrower.
dti: The debt-to-income ratio of the borrower (amount of debt divided 
                                               by annual income).
fico: The FICO credit score of the borrower.
days.with.cr.line: The number of days the borrower has had a credit line.
revol.bal: The borrower's revolving balance (amount unpaid at the end of
             the credit card billing cycle).
revol.util: The borrower's revolving line utilization rate 
        (the amount of the credit line used relative to total credit available)
inq.last.6mths: The borrower's number of inquiries by creditors
             in the last 6 months.
delinq.2yrs: The number of times the borrower had been 30+ days 
                past due on a payment in the past 2 years.
pub.rec: The borrower's number of derogatory public records (
            bankruptcy filings, tax liens, or judgments).


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sns.set_style("whitegrid")

data = pd.read_csv('data/classification/loan_data.csv')


def ratio_by_credit(colname):
    meet = data[data['credit.policy'] == 1].groupby(by=colname).count()['credit.policy']
    no_meet = data[data['credit.policy'] == 0].groupby(by=colname).count()['credit.policy']
    ratio = no_meet/(meet + no_meet)
    ratio.plot(kind='bar')
    return ratio

# =============================================================================
# Exploratory
# =============================================================================
data.head(1).T
data.dtypes  # One string
stats = data.describe().T

sns.countplot(data['credit.policy'])  # Unbalanced


# =============================================================================
# # Purpose column
# =============================================================================
sns.countplot('purpose', data=data, hue='credit.policy')
data['purpose'].value_counts()
data.groupby(by='purpose')['credit.policy'].mean()
data.groupby(by='purpose')['credit.policy'].mean()
# Calculating ratio of each categorie
ratio_by_credit('purpose')

dummies = pd.get_dummies(data['purpose'], drop_first=True)
data = pd.concat([data.drop('purpose', axis=1), dummies], axis=1)

# =============================================================================
# # int.rate column
# =============================================================================
data['int.rate'].value_counts()
sns.distplot(data['int.rate'], bins=40)

# =============================================================================
# # int.rate column
# =============================================================================
data['installment'].value_counts()
sns.distplot(data['installment'], bins=40)

# =============================================================================
# # log.annual.inccolumn
# =============================================================================
data['log.annual.inc'].value_counts()
sns.distplot(data['log.annual.inc'], bins=40)

# =============================================================================
# # dti  column
# =============================================================================
data['dti'].value_counts()
sns.distplot(data['dti'], bins=40)

# =============================================================================
# # fico  column
# =============================================================================
data['fico'].value_counts()
sns.distplot(data['dti'], bins=40)
plt.figure()
data[data['credit.policy'] == 0]['fico'].hist(bins=35, color='red',
                                              label='Credit policy=0',
                                              alpha=0.4)
data[data['credit.policy'] == 1]['fico'].hist(bins=35, color='blue',
                                              label='Credit policy=1',
                                              alpha=0.5)
plt.legend()
plt.xlabel("FICO")
# =============================================================================
# # days.with.cr.line  column
# =============================================================================
data['days.with.cr.line'].value_counts()
sns.distplot(data['days.with.cr.line'], bins=40)

# =============================================================================
# # revol.bal column
# =============================================================================
data['revol.bal'].value_counts()
sns.distplot(data['revol.bal'], bins=40)  # some high values
sns.boxplot(data['revol.bal'], orient='v')

# =============================================================================
# # revol.bal column
# =============================================================================
data['revol.util'].value_counts()
sns.distplot(data['revol.util'], bins=40)  # some high values
sns.boxplot(data['revol.bal'], orient='v')

# =============================================================================
# # inq.last.6mthscolumn
# =============================================================================
data['inq.last.6mths'].value_counts()
sns.distplot(data['revol.util'], bins=40)  # some unuseful values


# =============================================================================
# # delinq.2yrs  scolumn
# =============================================================================
data['delinq.2yrs'].value_counts()
sns.countplot(data['delinq.2yrs'])  # some unuseful values

# =============================================================================
# # pub.rec   scolumn
# =============================================================================
data['pub.rec'].value_counts()
sns.countplot(data['pub.rec'])  # some unuseful values
dummies = pd.get_dummies(data['pub.rec'], drop_first=True)
data = pd.concat([data.drop('pub.rec', axis=1), dummies], axis=1)


# =============================================================================
# Splitting data
# =============================================================================
# =============================================================================

y = data['not.fully.paid'].values
X = data.drop('not.fully.paid', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# =============================================================================
# Decision Tree Model
# =============================================================================
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =============================================================================
# RF Tree Model
# =============================================================================
rf_model = RandomForestClassifier(n_estimators=300)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

