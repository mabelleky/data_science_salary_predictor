import GlassdoorScraper as gs
import pandas as pd

path = "D:/DS_Projects/data_science_salary_predictor/chromedriver.exe"

dataFrame = gs.get_jobs("data scientist", 15, False, path, 15)