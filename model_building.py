# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 19:50:26 2020

@author: Mabel
referenced from Ken Jee - Github username: PlayingNumbers
url: https://github.com/PlayingNumbers/ds_salary_proj
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

dataFrame = pd.read_csv('D:/DS_Projects/data_science_salary_predictor/eda_data.csv')

# To-do list
# - choose relevant volumns
# - with categorical data, when building models make dummy variables or get dummy data
# - creating dummy data increases # columns which has implications on what models to use
# - create train test split so that our model generalizes well
# - multiple linear regression
# - lasso regression (used because dataset will be sparce from dummy variables, lasso regression can help us normalize that)
# - random forest (have a tree-based model comparing to linear models)
## - gradient boosted tree
## - support vector regression
# - tune models GridsearchCV
# - test ensembles

dataFrame = dataFrame.drop(columns=['Unnamed: 0'])

# - choose relevant volumns
dataFrame.columns
dataFrame_model = dataFrame[['avg_salary', 'Size','Type of ownership', 'Industry', 
                    'Sector', 'Revenue', 'Province', 'company_age', 'python_yn', 
                    'rstudio_yn', 'spark', 'aws', 'excel', 'job_simplified', 
                    'seniority', 'desc_length']]

# - get dummy data
dataFrame_dummy = pd.get_dummies(dataFrame_model)

#train test split
X = dataFrame_dummy.drop('avg_salary', axis=1)
y = dataFrame_dummy.avg_salary.values # .values creates an array as opposed to series from dataFrame_dummy.avg_salary

#test_size=0.2 means 80% in train set and 20% in test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression

# statsmodel ols regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()















