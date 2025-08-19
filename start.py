import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def displayBatColumn(df: pd.DataFrame):
    for col in df.columns:
        print(f"{col}")
        
def countMonthData(df: pd.DataFrame) -> pd.DataFrame:
    monthDataDict = {"month": [], "count": []}  # initialize keys with empty lists
    totalNumOFMonth = len(df['month'].value_counts())
    for i in range(totalNumOFMonth - 1):
        monthDataDict["month"].append(i)
        monthDataDict["count"].append(df.query(f'month == {i}').shape[0])
        
   
    return pd.DataFrame(monthDataDict)

BatDf= pd.read_csv('dataset1.csv')
RatDf = pd.read_csv('dataset2.csv')
Ratnpy = RatDf.to_numpy()
Batnp = BatDf.to_numpy()

print(displayBatColumn(BatDf))

monthDataDf = countMonthData(BatDf)

monthData = countMonthData(BatDf)["count"].to_numpy()

print("##############################################################")

#get the bin width per 1 month
binWidth = 1
binCount = monthData.shape[0] // binWidth
maxValue = monthData.max()
minValue = monthData.min()
rangeValue = maxValue - minValue
print("maxValue", maxValue)
print("minValue", minValue) 
print("rangeValue", rangeValue)
print("binCount", binCount)

# plot histogram of flight delay time distribution
plt.bar(monthDataDf['month'], monthDataDf['count'], color='blue', edgecolor='black')
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Number of Test Taken per Month')
plt.show()




