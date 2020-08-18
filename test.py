import GlassdoorScraper as gs
import pandas as pd

path = "D:/DS_Projects/data_science_salary_predictor/chromedriver.exe"

pd.set_option("display.max_columns", None)
dataFrame = gs.get_jobs("data scientist", 50, False, path, 15)
print(dataFrame)