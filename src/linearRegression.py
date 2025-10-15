import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cleaning data and Data wrangling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.bat import BatData as BatData
from utils.rat import RatData as RatData
from utils.barchart import BarChart as BarChart
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm


Bat = BatData()
Rat = RatData() 

bat_df = Bat.bat_data
df2 = Rat.rat_data

#print the food availability descriptive statistics when season is 0 and 1
print(df2[df2['season'] == 0]['food_availability'].describe())
print(df2[df2['season'] == 1]['food_availability'].describe())



# Convert numeric columns (in case of string types)
for col in ["food_availability", "bat_landing_number", "hours_after_sunset", "rat_minutes", 'season']:
    df2[col] = pd.to_numeric(df2[col], errors="coerce")

# Drop missing values
df2 = df2.dropna(subset=["food_availability", "bat_landing_number", "hours_after_sunset", "rat_minutes", 'season'])

# Define explanatory (X) and response (Y)
X = df2[["bat_landing_number", "hours_after_sunset", "rat_minutes", 'season']]
X = sm.add_constant(X)  # add intercept
y = df2["food_availability"]

# Fit multiple linear regression model
model = sm.OLS(y, X).fit()

# Display full summary
print(model.summary())


sns.pairplot(df2, x_vars=["bat_landing_number","hours_after_sunset","rat_minutes", 'season'], y_vars="food_availability", kind="reg")
plt.show()

#print correlation matrix in graphic form
plt.figure(figsize=(10, 8))
sns.heatmap(df2.iloc[:, [1,2, 3, 4, 5, 6, 9]].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()