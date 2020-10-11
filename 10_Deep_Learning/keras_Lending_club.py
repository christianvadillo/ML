# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:54:35 2020

@author: 1052668570

We will be using a subset of the LendingClub DataSet obtained from Kaggle:
    https://www.kaggle.com/wordsforthewise/lending-club

    LendingClub is a US peer-to-peer lending company, headquartered in
    San Francisco, California.[3] It was the first peer-to-peer lender
    to register its offerings as securities with the Securities and Exchange
    Commission (SEC), and to offer loan trading on a secondary market.
    LendingClub is the world's largest peer-to-peer lending platform.

Our Goal:
Given historical data on loans given out with information on whether or not
the borrower defaulted (charge-off), can we build a model that can predict
wether or nor a borrower will pay back their loan? This way in the future
when we get a new potential customer we can assess whether or not they are
likely to pay back the loan. Keep in mind classification metrics when
evaluating the performance of your model!

The "loan_status" column contains our label.

LoanStatNew	Description
0	loan_amnt	    The listed amount of the loan applied for by the borrower.
                    If at some point in time, the credit department reduces the
                    loan amount, then it will be reflected in this value.
1	term	        The number of payments on the loan. Values are in months
                    and can be either 36 or 60.
2	int_rate	    Interest Rate on the loan
3	installment	    The monthly payment owed by the borrower if the loan
                    originates.
4	grade	        LC assigned loan grade
5	sub_grade	    LC assigned loan subgrade
6	emp_title	    The job title supplied by the Borrower when applying for
                    the loan.*
7	emp_length	    Employment length in years. Possible values are between
                    0 and 10 where 0 means less than one year and 10 means
                    ten or more years.
8	home_ownership	The home ownership status provided by the borrower during
                    registration or obtained from the credit report.
                    Our values are: RENT, OWN, MORTGAGE, OTHER
9	annual_inc	    The self-reported annual income provided by the borrower
                    during registration.
10	verification_status	Indicates if income was verified by LC, not verified,
                        or if the income source was verified
11	issue_d	        The month which the loan was funded
12	loan_status	    Current status of the loan
13	purpose	        A category provided by the borrower for the loan request.
14	title	        The loan title provided by the borrower
15	zip_code	    The first 3 numbers of the zip code provided by the borrower
                    in the loan application.
16	addr_state	    The state provided by the borrower in the loan application
17	dti	            A ratio calculated using the borrower’s total monthly debt
                    payments on the total debt obligations, excluding mortgage
                    and the requested LC loan, divided by the borrower’s
                    self-reported monthly income.
18	earliest_cr_line	The month the borrower's earliest reported credit line
                        was opened
19	open_acc	    The number of open credit lines in the borrower's credit file.
20	pub_rec	        Number of derogatory public records
21	revol_bal	    Total credit revolving balance
22	revol_util	    Revolving line utilization rate, or the amount of credit
                    the borrower is using relative to all available revolving
                    credit.
23	total_acc	    The total number of credit lines currently in the
                    borrower's credit file
24	initial_list_status	The initial listing status of the loan. Possible values
                        are – W, F
25	application_type	Indicates whether the loan is an individual application
                        or a joint application with two co-borrowers
26	mort_acc	    Number of mortgage accounts.
27	pub_rec_bankruptcies	Number of public record bankruptcies

"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sns.set_style("whitegrid")


def get_info(col):
    return info.loc[col]['Description']


info = pd.read_csv(r"../Tenserflow/DATA/lending_club_info.csv", index_col=0)
data = pd.read_csv(r"../Tenserflow/DATA/lending_club_loan_two.csv")

data.head(1).T
data['loan_status'].unique()
sns.countplot(data['loan_status'])  # unbalanced problem
stats = data.describe().T

# =============================================================================
# # Correlation
# =============================================================================
sns.heatmap(data.corr(), annot=True, cmap='viridis')
plt.ylim(14, 0)

# Duplicate information between installment and loan_amnt
get_info('installment')
get_info('loan_amnt')
sns.scatterplot(x='installment', y='loan_amnt', data=data, hue='loan_status', alpha=0.5)
# It show a big correlation, we need to check this out later if the performance is lower

# Exists a relationship between loan_status and loan_amnt?
sns.boxplot(x='loan_status', y='loan_amnt', data=data)
data.groupby(by='loan_status')['loan_amnt'].describe().T
# avg of people with high loans tend to not pay


# =============================================================================
# # missing data?
# =============================================================================
data.info()  # mort_acc, emp_title, emp_length, title, revol_util, pub_rec_bankruptcies
sns.heatmap(data.isnull(), yticklabels=False, cmap='viridis', cbar=False)
data.isnull().sum()  # How many?
data.isnull().sum() / len(data) * 100  # What percentage?

# =============================================================================
# # emp_title have nearly 6% of missing data, which is a lot of data to drop. 
# # Then we can analyze if the feature give us relevant information for keep it
# =============================================================================
get_info('emp_title')
data['emp_title'].nunique()  # There are a lot of unique titles
data['emp_title'].value_counts() # to convert this to some sort of dummy feature
# We can apply extensive feature engineering, but we need expertise advice
# But for now we will remove this column because it do not gives us good information
data.drop('emp_title', axis=1, inplace=True)

# =============================================================================
# # emp_length have nearly 5% of missing data, which is a lot of data to drop. 
# # Then we can analyze if the feature give us relevant information for keep it
# =============================================================================
get_info('emp_length')
data['emp_length'].nunique()
data['emp_length'].value_counts() # to convert this to some sort of dummy feature
sorted(data['emp_length'].dropna().unique())
emp_l_sorted = ['< 1 year', '1 year', '2 years', '3 years', '4 years',
                '5 years', '6 years', '7 years', '8 years', '9 years',
                '10+ years']

sns.countplot('emp_length', data=data, hue='loan_status', order=emp_l_sorted)
# Check the relationship between Fully Paid and Charged Off for employment length
sns.countplot('emp_length', data=data, hue='loan_status', order=emp_l_sorted)
# If there a extreme difference in one of the categories then it's a fairly
# important feature
# If the ratio of Fully Paid to Charged Off is essentially the same across
# all these categories, then isn't a very informative feature
# Calculating ratio of each categorie
emp_co = data[data['loan_status'] == 'Charged Off'].groupby(by='emp_length').count()['loan_status']
emp_fp =data[data['loan_status'] == 'Fully Paid'].groupby(by='emp_length').count()['loan_status']
ratio = emp_co/ (emp_co + emp_fp)
ratio.plot(kind='bar')
# We can see that there looks extremely similar, no big differences between categories
# It doesn't matter how long a person works, close to 20% will not pay the loan
# Then for now we will remove this column because it do not gives us good information
data.drop('emp_length', axis=1, inplace=True)

# =============================================================================
# # purpose and title give us relevant information?
# get_info('purpose')
# =============================================================================
data['purpose'].nunique()  # There are a lot of unique titles
data['purpose'].value_counts() # to convert this to some sort of dummy feature
get_info('title')
data['title'].nunique()  # There are a lot of unique titles
data['title'].value_counts() # to convert this to some sort of dummy feature
# Because purpose and title have similar information, we can drop title column
data.drop('title', axis=1, inplace=True)

# =============================================================================
# # Looks that mort_acc have nearly 10% of missing data, which is a lot
# # of data to drop. Then we can drop off the entire column
# =============================================================================
get_info('mort_acc')
data['mort_acc'].nunique()  # There are a lot of unique titles
data['mort_acc'].value_counts() # to convert this to some sort of dummy feature
# Which other features are correlated with mort_acC?
data.corr()['mort_acc'].sort_values()
data['total_acc'].nunique()  # There are a lot of unique titles

mort_ac_median = data.groupby('total_acc').median()['mort_acc']  # mort_acc avg for total acc

def fill_mort_acc(columns):
    total_acc = columns[0]
    mort_acc = columns[1]
    if np.isnan(mort_acc):
        return mort_ac_median[total_acc]
    else:
        return mort_acc
    
data['mort_acc'] = data.apply(lambda x: fill_mort_acc(x[['total_acc', 'mort_acc']]), axis=1)

# =============================================================================
# # Droping the rest NaN
# =============================================================================
data = data.dropna()
data.isnull().sum()


# =============================================================================
# # Feature engineering for categorical variables
# =============================================================================
# =============================================================================
# =============================================================================
data.select_dtypes(['object']).columns  # Categorical columns

# =============================================================================
# # can we transform term?
# =============================================================================
get_info('term')
data['term'].value_counts()
# Transforming strings to 36 and 60
map_val = {' 36 months': 36, ' 60 months': 60}
data['term'] = data['term'].map(map_val)
sns.countplot('term', data=data, hue='loan_status')

# =============================================================================
# # Exploring grades and subgrades
# =============================================================================
get_info('grade')
get_info('sub_grade')

data['grade'].head()
data['grade'].value_counts()
sns.countplot('grade', data=data, hue='loan_status')

data['sub_grade'].head()
data['sub_grade'].value_counts()

subgrade_order = sorted(data['sub_grade'].unique())
sns.countplot('sub_grade', data=data, order=subgrade_order, palette='coolwarm')
sns.countplot('sub_grade', data=data, hue='loan_status', order=subgrade_order, palette='coolwarm')

f_and_g = data[(data['grade'] == 'G') | (data['grade'] == 'F')]
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot('sub_grade', data=f_and_g, hue='loan_status', order=subgrade_order)
# Because grade is subpart of subgrades, we can drop it
data.drop('grade', axis=1, inplace=True)
# Transforming subgrades to dummy variables
dummies = pd.get_dummies(data['sub_grade'], drop_first=True)
# Removing sub_grades and concatenating the rest columns with the dummies
data = pd.concat([data.drop('sub_grade', axis=1), dummies], axis=1)
# data.columns


# =============================================================================
# verification_status - transforming to dummy variable
# =============================================================================
get_info('verification_status')
data['verification_status'].value_counts()
dummies = pd.get_dummies(data['verification_status'], drop_first=True)
# Removing and concatenating the rest columns with the dummies
data = pd.concat([data.drop('verification_status', axis=1), dummies], axis=1)

# =============================================================================
# Transforming multiple categorical at once
# =============================================================================
get_info('application_type')
get_info('initial_list_status')
get_info('purpose')
data['initial_list_status'].value_counts()
dummies = pd.get_dummies(data[['application_type',
                          'initial_list_status',
                          'purpose']], drop_first=True)

data = pd.concat([data.drop(['application_type',
                          'initial_list_status',
                          'purpose'], axis=1), dummies], axis=1)


# =============================================================================
# Transforming home_ownership
# =============================================================================
get_info('home_ownership')
data['home_ownership'].value_counts()
# Merging OTHER, NONE and ANY
data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)
# Getting dummies
dummies = pd.get_dummies(data['home_ownership'], drop_first=True)
data = pd.concat([data.drop('home_ownership', axis=1), dummies], axis=1)

# =============================================================================
# address - analysis
# =============================================================================
data['address'].value_counts()
# Extracting zip codes
data['zip_code'] = data['address'].apply(lambda address: int(address[-5:]))
data['zip_code'].value_counts()
# Transforming to dummy variable
dummies = pd.get_dummies(data['zip_code'], drop_first=True)
data = pd.concat([data.drop(['zip_code', 'address'], axis=1), dummies], axis=1)

# =============================================================================
# issue_d - analysis
# =============================================================================
get_info('issue_d')
data['issue_d'].value_counts()
# dropping
data.drop('issue_d', axis=1, inplace=True)

# =============================================================================
# earliest_cr_line - analysis
# =============================================================================
data['earliest_cr_line'].value_counts()
# Extracting the year
data['earliest_cr_line'] = data['earliest_cr_line'].apply(lambda date: int(date[-4:]))

# =============================================================================
# Transforming y-label
# =============================================================================
get_info('loan_status')
data['loan_status'].head()
data['loan_status'].value_counts()
# Transforming strings to 0 and 1
# data['loan_status'] = data['loan_status'].apply(lambda y: 1 if "Fully" in y else 0)
data['loan_repaid'] = data['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

# Correlation with the target
data.corr()['loan_repaid'].sort_values(ascending=True)[:-1].plot(kind='bar')
data.drop('loan_status', axis=1, inplace=True)

# =============================================================================
# Samplind data (optional if we dont use gpu)
# =============================================================================
sampled_data = data.sample(frac=0.3, random_state=101)
# =============================================================================
# SPlitting data
# =============================================================================
X = sampled_data.iloc[:, :-1].values
y = sampled_data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# Creating the model
# =============================================================================
X_train.shape
model = Sequential()

model.add(Dense(78, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(21, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(x=X_train, y=y_train, epochs=25,
          batch_size=256, validation_data=(X_test, y_test),
           callbacks=[early_stopping])

# =============================================================================
# Evaluating model
# =============================================================================
loss_df = pd.DataFrame(model.history.history)   
loss_df.plot()

y_pred = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
