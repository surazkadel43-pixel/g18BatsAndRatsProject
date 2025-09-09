import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import statistics as stats
import scipy.stats as st
import matplotlib.pyplot as plt

from utils.bat import BatData
from utils.rat import RatData
 
df1 = BatData().bat_data
df2 = RatData().rat_data
 
# Histogram
plt.hist(df1["seconds_after_rat_arrival"], bins=20)
plt.xlabel("Seconds after rat arrival")
plt.ylabel("Number of landings")
plt.title("Distribution of seconds_after_rat_arrival")
plt.show() # The histogram shows that the data is skewed to the right which means we should use median instead of mean
 
# Summary stats
print(df1["seconds_after_rat_arrival"].describe())
 
# Alternate Hypothesis: The proportion of risk-taking bats is lower for early landings (they avoid risk more when rats just arrived).
# Null Hypothesis: The proportion of risk-taking bats is the same for early vs late landings.
 
# Define early vs late based on median split
cutoff = df1["seconds_after_rat_arrival"].median()
print("this is the median of seconds_after_rat_arrival:", cutoff)
 
df1["timing"] = np.where(df1["seconds_after_rat_arrival"] <= cutoff, "early", "late")
 
# Split the risk data into two groups
early_risk = df1[df1["timing"] == "early"]["risk"]
late_risk = df1[df1["timing"] == "late"]["risk"]
 
# Descriptive stats
print("Early mean risk-taking:", early_risk.mean())
print("Late mean risk-taking:", late_risk.mean())
 
# Two-sample t-test (one-sided: early < late)
t_stats, p_val = st.ttest_ind(early_risk, late_risk, alternative="less")
 
print("\nT-statistic:", t_stats)
print("P-value:", p_val)
 
if p_val <= 0.05:
    print("Reject H0: Early risk-taking is significantly lower (supports predator perception).")
else:
    print("Fail to reject H0: No evidence of predator perception.")
    