import pandas as pd
import matplotlib.pyplot as plt


BatDf = pd.read_csv("dataset1.csv")
RatDf = pd.read_csv("dataset2.csv")


aprilBatDf = BatDf[BatDf["month"] == 4]




