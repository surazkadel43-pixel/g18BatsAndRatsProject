import sys, os
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Cleaning data and Data wrangling
from utils.bat import BatData as BatData
from utils.rat import RatData as RatData
from utils.barchart import BarChart as BarChart


bat = BatData()
rat = RatData() 
bat.merger_dataset_1_2_on_rat_period_start()


batDf = bat.bat_data
ratDf = rat.rat_data 

print(bat.applyRegressionModel())
batDf = bat.updated_bat_data()
# bat.showScatterPlot(season= 0)
# bat.showScatterPlot(season= 1)
bat.multiple_linear_regression_of_bat_landing(season=0)
bat.multiple_linear_regression_of_bat_landing(season=1)

# #when season is 1
# print(batDf.info())

# print(batDf[batDf['season'] == 1][['bat_landing_to_food','seconds_after_rat_arrival', 'predicted_food_availability', 'rat_minutes', 'season', 'actual_hours_after_sunset']].describe())
# print(batDf[batDf['season'] == 0][['bat_landing_to_food','seconds_after_rat_arrival', 'predicted_food_availability', 'rat_minutes', 'season', 'actual_hours_after_sunset']].describe())
# print(batDf[['bat_landing_to_food','seconds_after_rat_arrival', 'predicted_food_availability', 'rat_minutes', 'season', 'actual_hours_after_sunset']].describe())

# corelation matrix in chart form
correlation_matrix = batDf[['bat_landing_to_food','seconds_after_rat_arrival', 'predicted_food_availability', 'rat_minutes', 'season', 'actual_hours_after_sunset','hour']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()