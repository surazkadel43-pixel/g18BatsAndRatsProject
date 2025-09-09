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
rat_df = RatData(4) # April

bat_df.order_data_by_time()
rat_df.count_food_availability_per_hour()
rat_df.count_rat_arrivals_per_hour()

BarChart.plotBarChart(rat_df.rat_data, 'hour', 'avg_food_availability_hourly', 'Food Availability by Hour', 'Hour', 'Food Availability', "green")
BarChart.plotBarChart(rat_df.rat_data, 'hour', 'avg_rat_arrivals_hourly', 'Rat Arrivals by Hour', 'Hour', 'Rat Arrivals', "red")

# analyse the data bat landing to food platform with when number of rat arrivals is greater than 1
# analyse how they approach food ( bat timing to approach food platform )
