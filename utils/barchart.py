import matplotlib.pyplot as plt
import pandas as pd

class BarChart:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.bar(self.data['hour'], self.data['count'])
        plt.xlabel('Hour')
        plt.ylabel('Count')
        plt.title('Bat Activity by Hour')
        plt.xticks(rotation=45)
        plt.show()
    
    @staticmethod
    def plotBarChart(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, color: str = "skyblue"):
        """
        Plots a bar chart of a specified column in a DataFrame.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(df[x], df[y], color=color, edgecolor='black')
        plt.title(title)
        plt.xticks(df[x])   # or range(24) if you want fixed ticks
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
