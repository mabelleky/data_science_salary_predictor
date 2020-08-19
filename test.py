"""
Created on Sat Aug 17  05:28:31 2020

@author: mabelleky
"""

import GlassdoorScraper as gs
import pandas as pd

path = "D:/DS_Projects/data_science_salary_predictor/chromedriver.exe"

pd.set_option("display.max_columns", None)
dataFrame = gs.get_jobs("data scientist", 30, False, path, 15)
# dataFrame.to_csv("glassdoor_jobs.csv", index=False)
print(dataFrame)