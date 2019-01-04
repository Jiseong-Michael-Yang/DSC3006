# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:16:56 2018

@author: Jiseong Yang
"""

#%% Data Preprocessing

# Import appropriate modules
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Set the working directory
os.getcwd()
wd = r"C:\Users\Jiseong Yang\git_projects\DSC3006\Assignment\Assignment1"
wd = wd.replace("'\'", "/")
os.chdir(wd)

#%% Load the datasets
x_train = pd.read_csv("X_train.csv")
x_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")

# Temporarilly remove "id" column
id_train = pd.DataFrame(x_train.id)
x_train = x_train.drop(columns="id")
id_test = pd.DataFrame(x_test.id)
x_test = x_test.drop(columns="id")

#%% Missing value check
True in x_train.isnull()
True in x_test.isnull()
True in y_train.isnull()

#%% Categorical data encoding
## Mapping ordinal features 
size_mapping = {
        "LE3": 0,
        "GT3": 1
        }

x_train["famsize"] = x_train["famsize"].map(size_mapping)
x_test["famsize"] = x_test["famsize"].map(size_mapping)

#%% Performing one-hot encoding on nominal features
## Mapping binary features
nominal_features = ['school', 'sex', 'address', 'Pstatus', 'schoolsup',
                    'famsup', 'paid', 'activities', 'nursery', 'higher',
                    'internet', 'romantic', 'Mjob', 'Fjob', 'reason', 
                    'guardian' ]

numerical_features = list(set(x_train.columns) - set(nominal_features))

x_train_dummies = pd.get_dummies(x_train[nominal_features], drop_first=True)
x_test_dummies = pd.get_dummies(x_test[nominal_features], drop_first=True)

#%% Scaling: Standardization
# Extractinig numerical features
x_train_numeric = x_train[numerical_features]
x_test_numeric = x_test[numerical_features]

# Standardization
stds = StandardScaler()
stds.fit(x_train_numeric)
x_train_stds = pd.DataFrame(stds.transform(x_train_numeric))
x_test_stds = pd.DataFrame(stds.transform(x_test_numeric))

# Reassigning column names
x_train_stds.columns = numerical_features
x_test_stds.columns = numerical_features

#%% Combining Datasets
x_train = pd.concat([id_train, x_train_stds, x_train_dummies], axis=1)
x_test = pd.concat([id_test, x_test_stds, x_test_dummies], axis=1)

#%% Classification: Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Fitting the model
y_train_result_only = y_train["G3C"]
gnb = GaussianNB()
gnb.fit(x_train, y_train_result_only)

# Make predictions
y_pred = pd.DataFrame(gnb.predict(x_test))
y_pred.columns = ["Category"]

# Make submission files
answer = pd.concat([x_test["id"], y_pred], axis=1)
answer.to_csv("NB_result.csv", index=None)

