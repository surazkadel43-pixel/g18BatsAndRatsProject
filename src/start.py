import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cleaning data and Data wrangling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.bat import BatData as BatData
from utils.rat import RatData as RatData
from utils.barchart import BarChart as BarChart

bat_df = BatData()
rat_df = RatData() 

bat_df.risk_reward_distribution()



print(bat_df.risk_reward_confidence_interval())

# bat_df.order_data_by_time()
bat_df.merger_dataset_1_2_on_rat_period_start()

grouped_stats = bat_df.bat_data.groupby('rat_arrival_number').agg({
    'time': 'count',
    'bat_landing_to_food': 'mean',
    'seconds_after_rat_arrival': 'mean',
    'bat_landing_number': 'mean',
    'food_availability': 'mean'
}).reset_index()
print(grouped_stats)
rat_df.count_food_availability_per_hour()
rat_df.count_rat_arrivals_per_hour()
rat_df.correlation_rat_food()
rat_df.correlation_rat_bats()
print(rat_df.regression_food_on_rats())
print(rat_df.regression_bat_on_rats_food())
print(rat_df.summarize_bat_food_by_rat_presence())



# BarChart.plotBarChart(rat_df.rat_data, 'hour', 'avg_food_availability_hourly', 'Food Availability by Hour', 'Hour', 'Food Availability', "green")
# BarChart.plotBarChart(rat_df.rat_data, 'hour', 'avg_rat_arrivals_hourly', 'Rat Arrivals by Hour', 'Hour', 'Rat Arrivals', "red")


