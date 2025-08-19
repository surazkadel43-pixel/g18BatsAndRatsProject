import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.proportion as stm 
import scipy as sp

BatDf = pd.read_csv('dataset1.csv')
RatDf = pd.read_csv('dataset2.csv')

def get24HourTimelineForMonth(df: pd.DataFrame, month: int) -> pd.Series:
    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Ensure start_time is datetime
    df['start_time'] = pd.to_datetime(df['start_time'], format="%d/%m/%Y %H:%M", errors='coerce')

    # Filter by the month column
    monthly_data = df.loc[df['month'] == month].copy()

    # Create 'hour' column
    monthly_data.loc[:, 'hour'] = monthly_data['start_time'].dt.hour

    # Count events per hour and fill missing hours with 0
    return monthly_data.groupby('hour').size().reindex(range(24), fill_value=0)

def get24HourTimeLineWithRiskAndReward(df: pd.DataFrame, risk: int, reward: int, month: int) -> pd.Series:
    
    #monthly_data
    monthly_data = df[df['month'] == month]
    
    #get the the data with risk and reward
    monthly_data = monthly_data[(monthly_data['risk'] == risk) & (monthly_data['reward'] == reward)]
    
    
    # Create 'hour' column
    monthly_data.loc[:, 'hour'] = pd.to_datetime(monthly_data['start_time'], format="%d/%m/%Y %H:%M", errors='coerce').dt.hour
    
    # Count events per hour and fill missing hours with 0
    return monthly_data.groupby('hour').size().reindex(range(24), fill_value=0)

def getConfidenceLevel(df:pd.DataFrame, firstData: int, secondData: int, conf_lvl: float = 0.95) -> tuple:
    """
    Calculate the confidence interval for the proportion of a specific event in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    firstData (int): The count of the first event.
    secondData (int): The count of the second event.
    conf_lvl (float): The confidence level for the interval.
    
    Returns:
    tuple: Lower and upper bounds of the confidence interval.
    """
    total = firstData + secondData
    sig_lvl = 1 - conf_lvl
    ci_low, ci_upp = stm.proportion_confint(firstData, total, alpha=sig_lvl, method="normal")
    return ci_low, ci_upp





#plot risk and reward
# risk = 0
# reward = 0
# risk_reward_counts = get24HourTimeLineWithRiskAndReward(BatDf, risk, reward, month=5)
# print(risk_reward_counts, risk_reward_counts.sum()) 
# risk_reward_counts.plot(kind='bar', figsize=(10,5))
# plt.xlabel("Hour of Day")
# plt.ylabel("Count")
# plt.title(f"Bat Activity with Risk {risk} and Reward {reward} in 24-Hour Timeline Total Count: {risk_reward_counts.sum()}")
# plt.show()


# get confidence level 
filteredonth = BatDf[BatDf['month'] == 4]
filterdData = filteredonth[filteredonth['month'] == 4][~ ((filteredonth['risk'] == 0) & (filteredonth['reward'] == 0))]



hardway = filterdData.query("risk == 1 and reward == 1")

easyway = filterdData.query("~(risk == 1 and reward == 1)")

print("easyway", easyway.shape[0])
print("hardway", hardway.shape[0])


print(filterdData.shape[0])


ci_low, ci_upp = getConfidenceLevel(filterdData,easyway.shape[0], hardway.shape[0],  conf_lvl=0.95)


print("C.I. of proportion at 95 percent confidence level is %.3f  and %.3f . for the bat going wasy way" % 
      ( ci_low*100,  ci_upp*100))

print("It means that the bat has a %.3f%% to %.3f%% chance of going easy way . so they want to avoid the bats" %
      (ci_low*100, ci_upp*100))