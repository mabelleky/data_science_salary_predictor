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

# statsmodel ols regression - applied to all data
"""
when using statsmodel, you have to create a constant and a constant creates
the constant in the model.  When you have a regression, you're fitting a line
to data, you have the slope of the line, but also have the intercept
and a column of all ones creates that intercept in statsmodels
"""
import statsmodels.api as sm
# model applied to all data
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
print(model.fit().summary())

"""
R-squared: 1.000 means the model explains about 100% of the variation in
glassdoor salaries, R-squared is 0.7 means the model explains about 70% of the
variation in glassdoor salaries

P>|t| ---> p-value of less than 0.05 means its significant in our model

coef ---> ex. looking at coefficients having the Python skill tends to pay more
ex2. Province-wise, BC pays more than Ontario

"""

# sklearn linear model with cross validation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
# don't have to fit for the approach we're taking because we'll use cross validation
"""
sklearn cross val score - take a sample data and a validation set and will 
run the model on the sample and evaluate on the validation set that's held
out to see if the model generalizes
- similar to a train test split that gives us how the model is performing
in reality
"""

lm = LinearRegression()
lm.fit(X_train, y_train)

#negative mean absolute error for this may be the most representative
#this will show how far on average off of our general prediction
#ex. off by 21 would mean the average is off by $21,000
cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error')
#cross validation = 3
cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3) 
np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3))

"""
Results from our data:
array([-10.3721043 , -13.62316551,  -9.24052134, -14.02304655, -20.65463737])

Taking the mean of the cross validation of 3, we have benchmark of $17,000 off
"""

"""
Note: When your matrix is sparse, it can be hard to get good values from
multiple linear regression because of the limited data.
We can address this with lasso regression since it normalizes 
the valies and would be better for our model
"""

# lasso regression
































