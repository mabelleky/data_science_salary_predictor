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

# - CHOOSE RELEVENT VOLUMES
dataFrame.columns
dataFrame_model = dataFrame[['avg_salary', 'Size','Type of ownership', 'Industry', 
                    'Sector', 'Revenue', 'Province', 'company_age', 'python_yn', 
                    'rstudio_yn', 'spark', 'aws', 'excel', 'job_simplified', 
                    'seniority', 'desc_length']]

# - GET DUMMY DATA
dataFrame_dummy = pd.get_dummies(dataFrame_model)

# TRAIN TEST SPLIT
X = dataFrame_dummy.drop('avg_salary', axis=1)
y = dataFrame_dummy.avg_salary.values 
# .values creates an array as opposed to series from dataFrame_dummy.avg_salary

#test_size=0.2 means 80% in train set and 20% in test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# MULTIPLE LINEAR REGRESSION

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


# sklearn LINEAR MODEL WITH CROSS VALIDATION
from sklearn.linear_model import LinearRegression, Lasso
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
"""
Results from our data:
array([-10.3721043 , -13.62316551,  -9.24052134, -14.02304655, -20.65463737])

Taking the mean of the cross validation of 3, we have benchmark of $17,000 off
"""
np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error'))
# RESULTS -13.582695014639645


#cross validation = 3
#cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3) 
#np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3))

"""
Note: When your matrix is sparse, it can be hard to get good values from
multiple linear regression because of the limited data.
We can address this issue with lasso regression since it normalizes 
the values and would be better for our model
"""


# LASSO REGRESSION
"""
Normalization term alpha: if alpha is 0 it's the same thing as the ols
multiple linear regression.  As we increase alpha, it increases the amount
that the data is smooth

Note that alpha in sklearn lasso regression defaults to 1
"""
lm_l = Lasso()
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring = 'neg_mean_absolute_error'))
#Results is -10.440155794581443, so about $10,000 meaning it's doing better

#If the results show that it starts off worse thougn, we can try different values
#using method below
alpha = []
error = []

for i in range(1, 100):
    alpha.append(i/10) #if data tapers off in higher #s try alpha.append(i/100)
    lml = Lasso(alpha=(i/10)) #lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lm_l, X_train, y_train, 
                                         scoring = 'neg_mean_absolute_error')))

plt.plot(alpha, error)

"""
RESULTS: Came across with smallest eigenvalue 1.09e-23 indicating a strong
multicollinearity problems or design matrix is singular. 
Multicollinearity is a problem because it undermines the statistical significance
of an independent variable. The larger the standard error of a regression 
coefficient, less likely it is that this coefficient will be statistically significant
"""

# if you have a peak, below ties it into a tuple
err = tuple(zip(alpha,error))

# with best error term (if it's significant) , we can use it to bench mark
# against our normal regression with a not so ideal error term
dataFrame_error = pd.DataFrame(err, columns = ['alpha', 'error'])
dataFrame_error[dataFrame_error.error == max(dataFrame_error.error)]
# Note: model turning in general could be improved with a grid search



# RANDOM FOREST
""" 
we would expect random forest to perform well here because the tree-based
decision process with binary values.  Also, do not have to worry as much about
multicollinearity with this type of model
"""
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error'))
# RESULTS  -9.34291758241758

# next is TUNE MODELS WITH GridsearchCV
""" what grid search does is you put in the parameters that you want and
it runs all models and spits out the one with best results
"""
from sklearn.model_selection import GridSearchCV
# focusing on depth and criterion below, but you can tweak other parameters on
# GridSearchCV documentation so your model can generalize better
"""a couple of approaches: 
    1) normal grid search - exhaustive method going through all scenarios, which
    has a factorial of complexity.
    
    2) randomize grid search - if you're limited in time, you can do a randomized
    grid search which takes a sample 
"""

parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'),
              'max_features':('auto', 'sqrt', 'log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error')
gs.fit(X_train,y_train)

gs.best_score_
# -8.798186813186813 - Performed better then linear regression and slightly better than Random Forest
gs.best_estimator_ 
# shows the parameters of that model, RESULTS: # of estimators at 10 was the best


# TEST ENSEMBLES
# using these different models to predict test set data if we get similar results

tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm) #RESULTS 7.382240523891751
mean_absolute_error(y_test,tpred_lml) #RESULTS 8.370914825592601
mean_absolute_error(y_test,tpred_rf) #RESULTS 4.1800000000000015
# rf model is shown to be performing the best compared to the lm and lml

"""sometimes it might make sense to combine acouple of different models 
and see if you can improve the performance through that
"""
mean_absolute_error(y_test,(tpred_lm + tpred_rf)/2)
# RESULTS 4.924302722450286

"""
might be able to run (tpred_lm + tpred_rf)/2 through another regression models
and get actual weights associated with it
Exploration consideration: Take ratio of 90% from random forest 
and 10% from other models
"""

























