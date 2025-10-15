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


bat.showScatterPlot(season= 0)
bat.showScatterPlot(season= 1)
bat.showScatterPlot(season= 2)
bat.multiple_linear_regression_of_bat_landing(season=0)
bat.multiple_linear_regression_of_bat_landing(season=1)
bat.multiple_linear_regression_of_bat_landing(season=2)


#########Rat Analysis#########
# Correlation Matrix
rat.showScatterPlot(season= 0,rat_presence=0)
rat.showScatterPlot(season= 1,rat_presence=0)
rat.showScatterPlot(season= 2,rat_presence=0)

# Multiple Linear Regression for y_col = 'bat_landing_number', x_col = [ 'food_availability', 'rat_minutes', 'season', 'hours_after_sunset']):

rat.build_and_evaluate_regression_using_z_stand_rescale(season=0,rat_presence=0)
rat.build_and_evaluate_regression_using_z_stand_rescale(season=1,rat_presence=0)
rat.build_and_evaluate_regression_using_z_stand_rescale(season=2,rat_presence=0)

# Multiple Linear Regression for y_col = 'food_availability', x_col = [  'rat_minutes', 'season', 'hours_after_sunset']):
rat.build_and_evaluate_regression_using_z_stand_rescale(y_col='food_availability', x_col=['rat_minutes', 'season', 'hours_after_sunset'], season=0,rat_presence=0)
rat.build_and_evaluate_regression_using_z_stand_rescale(y_col='food_availability', x_col=['rat_minutes', 'season', 'hours_after_sunset'], season=1,rat_presence=0)
rat.build_and_evaluate_regression_using_z_stand_rescale(y_col='food_availability', x_col=['rat_minutes', 'season', 'hours_after_sunset'], season=2,rat_presence=0)