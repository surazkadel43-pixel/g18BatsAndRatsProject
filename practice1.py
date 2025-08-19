import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BatDf = pd.read_csv('dataset1.csv')
RatDf = pd.read_csv('dataset2.csv')

print(BatDf[['start_time','bat_landing_to_food','rat_period_start','rat_period_end','seconds_after_rat_arrival', 'risk', 'reward']].head(60))


batLandingtoFoodApril = BatDf.query(" month == 4")['bat_landing_to_food']
batSecondsAfterRatArrivalApril = BatDf.query(" month == 4")['seconds_after_rat_arrival']
rewardCountWhenThereIsRisk = BatDf.query("risk == 0 and reward == 0  and month == 4")

print("batLandingtoFoodApril", batLandingtoFoodApril.describe())
print("batSecondsAfterRatArrivalApril", batSecondsAfterRatArrivalApril.describe())
print("rewardCountWhenThereIsRisk", rewardCountWhenThereIsRisk)
