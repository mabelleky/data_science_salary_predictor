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
# remove rows with -1 (missing values) for Salary Estimates - DONE
# parsing out salaries with lambda functions - DONE
# Removing numbers for Company Name, text only - DONE
# Age of Company - DONE
# add Province column for Location (Containing city names)
# parsing thru job description (python, etc.)

"""
Note to self: lambda - small anonymous function
lambda args : expression

Ex. x = lamda a, b, c : a + b + c
print(x(5, 6, 2))
13
"""

# parsing out salaries

dataFrame = dataFrame[dataFrame['Salary Estimate'] != '-1']
salary = dataFrame['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kDInSalary = salary.apply(lambda x: x.replace('k','').replace('$','')
                                .replace('CA',''))

dataFrame['min_salary'] = minus_kDInSalary.apply(lambda x: int(x.split('-')[0]))
dataFrame['max_salary'] = minus_kDInSalary.apply(lambda x: int(x.split('-')[1]))
dataFrame['avg_salary'] = (dataFrame.min_salary+dataFrame.max_salary)/2


# parsing company names

dataFrame['company_name_text'] = dataFrame.apply(lambda x: x['Company Name']
                                                 if x['Rating'] < 0
                                                 else x['Company Name'][:-3],
                                                 axis = 1)

# Age of Company

dataFrame['company_age'] = dataFrame['Founded'].apply(lambda x: x if x < 0
                                                      else 2020 - x)

# add Province column for Locations
switchCase = {
    'calgary': 'AB',
    'edmonton': 'AB',
    'burnaby': 'BC',
    'vancouver': 'BC',
    'brampton': 'ON',
    'markham': 'ON',
    'missisauga': 'ON',
    'ottawa': 'ON',
    'toronto': 'ON',
    'waterloo': 'ON',
    'regina': 'SK',
    'montreal': 'QC',
    'saint-laurent': 'QC',
    }

def checkProvince(cityName):
    return switchCase.get(cityName.lower(), "-1")

dataFrame['Province'] = dataFrame['Location'].apply(lambda x: checkProvince(x))

# parsing thru job descriptions




"""
Extra code for parsing out salaries
If data includes values 'per hour' and 'employer provided salary' in
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

If not in Spyder IDE, can check for type in console
Ex. dataFrame['min_salary'].dtype
    
"""

