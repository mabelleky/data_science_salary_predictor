# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 17:05:20 2020

@author: Mabel
referenced from Ken Jee - Github username: PlayingNumbers
"""

import pandas as pd

pd.set_option('display.max_columns', None) # display all columns in Pycharm
dataFrame = pd.read_csv('glassdoor_jobs.csv')

#To-Do List
# remove rows with -1 (missing values) for Salary Estimates
# parsing out salaries with lambda functions
# Removing numbers for Company Name, text only
# age of company
# add Province column for Location (Containing city names)
# parsing thru job description (python, etc.)

"""
Note to self: lambda like 
def func(x,y):
    some code 
"""

dataFrame = dataFrame[dataFrame['Salary Estimate'] != '-1']
salary = dataFrame['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kDInSalary = salary.apply(lambda x: x.replace('k','').replace('$','')
                                .replace('CA',''))




"""
If data includes values 'per hour' and/or 'employer provided salary' in
Salary estimate column, create a separate column for each

dataFrame['hourly'] = dataFrame['Salary'].apply(lambda x: 1 if 'per hour' in 
                        x.lower() else 0)
dataFrame['employer_provided'] = dataFrame['Salary Estimate'].apply(lambda x: 
                        1 if 'employer provided salary:' in x.lower() else 0)

min_hr = minus_kDInSalary.apply(lambda x: x.lower().replace('per hour', '')
                        .replace('exployer provided salary',''))
dataFrame['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
dataFrame['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
dataFrame['avg_salary'] = (dataFrame.min_salary+dataFrame.max_salary)/2

If not in Spyder IDE, can check for type
Ex. dataFrame['min_salary'].dtype
    
"""