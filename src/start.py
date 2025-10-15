import sys, os
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cleaning data and Data wrangling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.bat import BatData as BatData
from utils.rat import RatData as RatData
from utils.barchart import BarChart as BarChart

bat = BatData()
rat = RatData() 

# bat_df.risk_reward_distribution()
# bat_df.analyze_risk_vs_landing()


# print(bat_df.risk_reward_confidence_interval())

# # bat_df.order_data_by_time()
# bat_df.merger_dataset_1_2_on_rat_period_start()

# grouped_stats = bat_df.bat_data.groupby('rat_arrival_number').agg({
#     'time': 'count',
#     'bat_landing_to_food': 'mean',
#     'seconds_after_rat_arrival': 'mean',
#     'bat_landing_number': 'mean',
#     'food_availability': 'mean'
# }).reset_index()
# print(grouped_stats)
# rat_df.count_food_availability_per_hour()
# rat_df.count_rat_arrivals_per_hour()
# rat_df.correlation_rat_food()
# rat_df.correlation_rat_bats()
# print(rat_df.regression_food_on_rats())
# print(rat_df.regression_bat_on_rats_food())
# print(rat_df.summarize_bat_food_by_rat_presence())

bat.merger_dataset_1_2_on_rat_period_start()
bat_df = bat.bat_data
rat_df = rat.rat_data



#multiple linear regression modal reponse variable ( seconds_after_rat_arrival )

y = bat_df['seconds_after_rat_arrival'].values.reshape(-1,1)
x = bat_df[['rat_arrival_number', 'rat_minutes','risk','reward','hour']].values
#split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)
print("x_train size:", x_train.size)
print("x_test size:", x_test.size)
model = LinearRegression()
model.fit(x_train, y_train)
# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficients: ", model.coef_)
# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(x_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test.flatten(),"Predicted": y_pred.flatten(), "Input": x_test.tolist()})
print(df_pred)


# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred)

print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)

"""
COMPARE THE PERFORMANCE OF THE LINEAR REGRESSION MODEL
VS.
A DUMMY MODEL (BASELINE) THAT USES MEAN AS THE BASIS OF ITS PREDICTION
"""

# Compute mean of values in (y) training set
y_base = np.mean(y_train)

# Replicate the mean values as many times as there are values in the test set
y_pred_base = [y_base] * len(y_test)





# Compute standard performance metrics of the baseline model:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred_base)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred_base)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred_base))

# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred_base)

print("Baseline performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)



subset = bat_df[bat_df['rat_arrival_number'] == 1]
X = sm.add_constant(subset[['rat_minutes']])
y = subset['seconds_after_rat_arrival']

 
model = sm.OLS(y, X).fit()
print(model.summary())

