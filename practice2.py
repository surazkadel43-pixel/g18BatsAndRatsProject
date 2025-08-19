import pandas as pd

BatDf = pd.read_csv("dataset1.csv")
RatDf = pd.read_csv("dataset2.csv")

# Make sure they don't clash on 'time' vs 'start_time'
RatDf = RatDf.rename(columns={"time": "start_time_rat"})

# Concatenate side by side
combined_df = pd.concat([BatDf, RatDf], axis=1)

print(combined_df[['start_time','start_time_rat']].head(10))

