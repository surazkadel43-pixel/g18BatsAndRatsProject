import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
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
    
    @staticmethod
    def plotCorrelation(df: pd.DataFrame, x: str, y: str, title: str = None):
        """
        Plots correlation between two columns of a DataFrame using a scatter plot + regression line.
        """
        # Compute correlation
        corr = df[[x, y]].corr().iloc[0, 1]
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.regplot(
            x=x,
            y=y,
            data=df,
            scatter_kws={'s': 60, 'alpha': 0.7},
            line_kws={'color': 'red', 'lw': 2}
        )
        
        # Add title with correlation value
        if not title:
            title = f"Correlation between {x} and {y}"
        plt.title(f"{title}\nPearson r = {corr:.3f}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(alpha=0.3)
        plt.show()
        
        return round(float(corr), 3)  # return correlation value
    
    @staticmethod
    def plotPresenceBoxplot(df, zero_inflated_col, numeric_col):
        """
        Plots a boxplot for presence/absence analysis of a zero-inflated variable.
        - df: DataFrame
        - zero_inflated_col: column with mostly zeros (e.g., 'rat_minutes')
        - numeric_col: numeric column to compare (e.g., 'food_availability')
        """
        # Create presence/absence column
        presence_col = f"{zero_inflated_col}_present"
        df[presence_col] = (df[zero_inflated_col] > 0).astype(int)
        
        # Plot
        plt.figure(figsize=(8,6))
        sns.boxplot(x=presence_col, y=numeric_col, data=df)
        plt.xlabel(f"{zero_inflated_col} Present (0=No, 1=Yes)")
        plt.ylabel(numeric_col)
        plt.title(f"{numeric_col} when {zero_inflated_col} is Present vs Absent")
        plt.show()
    
    @staticmethod
    def plot_regression(df, x_col, y_col, regression_result):
        """
        Scatter plot with regression line.

        Parameters:
        - df: DataFrame containing the data
        - x_col: predictor column name (e.g., 'rat_arrival_number')
        - y_col: response column name (e.g., 'food_availability')
        - regression_result: dict with keys 'intercept' and 'slope_rat_arrival'
        """
        x = df[x_col]
        y = df[y_col]

        # Optional: jitter for heavily zero-inflated predictor
        if (x == 0).mean() > 0.8:
            x_plot = x + np.random.uniform(-0.1, 0.1, size=len(x))
        else:
            x_plot = x

        # Predicted values from regression
        y_pred = regression_result['intercept'] + regression_result['slope_rat_arrival'] * x

        # Plot
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=x_plot, y=y, alpha=0.5, s=40)
        plt.plot(x, y_pred, color='red', label=f"y = {regression_result['intercept']:.2f} + {regression_result['slope_rat_arrival']:.3f}*x")
        
        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel(y_col.replace("_", " ").title())
        plt.title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
        plt.legend()
        plt.show()
        
    @staticmethod
    def plot_risk_reward_chart(results):
        """
        Plots a bar chart of bat risk–reward strategies with confidence intervals.

        Parameters:
        - results: dict output from `risk_reward_confidence_interval`
        Example:
        {
            'easy_way': {'count': 386, 'proportion': 0.797, 'confidence_interval': (0.76, 0.83)},
            'hard_way': {'count': 98, 'proportion': 0.203, 'confidence_interval': (0.17, 0.24)}
        }
        """
        labels = list(results.keys())
        proportions = [results[label]['proportion'] for label in labels]
        ci_lower = [results[label]['confidence_interval'][0] for label in labels]
        ci_upper = [results[label]['confidence_interval'][1] for label in labels]

        # Calculate error bars: distance from proportion to CI
        error = [[prop - low for prop, low in zip(proportions, ci_lower)],
                [up - prop for prop, up in zip(proportions, ci_upper)]]

        # Plot
        plt.figure(figsize=(6,5))
        bars = plt.bar(labels, proportions, yerr=error, capsize=5, color=['skyblue', 'salmon'])
        
        # Optional: annotate counts on top of bars
        for bar, label in zip(bars, labels):
            height = bar.get_height()
            count = results[label]['count']
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, str(count),
                    ha='center', va='bottom', fontsize=10)

        plt.ylabel("Proportion of bats")
        plt.title("Bat Risk–Reward Strategy (with 95% CI)")
        plt.ylim(0,1)
        plt.show()
    
    @staticmethod
    def plot_rat_bat_interactions(df):
        """
        Plots multiple visualizations to explore rat-bat interactions.

        Parameters:
        - df: DataFrame with columns
            'rat_arrival_number', 'bat_landing_number',
            'bat_landing_to_food', 'seconds_after_rat_arrival',
            'food_availability'
        """
        # Scatter: Bat landings vs Rat arrivals
        plt.figure(figsize=(7,5))
        sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', data=df, s=50, color='skyblue')
        plt.xlabel("Rat Arrivals")
        plt.ylabel("Bat Landings Number ")
        plt.title("Bat Landings vs Rat Arrivals")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        # Scatter: Bat landing timing vs Rat arrivals
        plt.figure(figsize=(7,5))
        sns.scatterplot(x='rat_arrival_number', y='seconds_after_rat_arrival', data=df, s=50, color='salmon')
        plt.xlabel("Rat Arrivals")
        plt.ylabel("Seconds After Rat Arrival")
        plt.title("Bat Landing Timing vs Rat Arrivals")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
        
        # Scatter: bat Landing to food vs Rat arrivals
        plt.figure(figsize=(7,5))
        sns.scatterplot(x='rat_arrival_number', y='bat_landing_to_food', data=df, s=50, color='salmon')
        plt.xlabel("Rat Arrivals")
        plt.ylabel("Bat Landing to Food")
        plt.title("Bat Landing to Food vs Rat Arrivals")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        # Scatter: Food availability vs Rat arrivals
        plt.figure(figsize=(7,5))
        sns.scatterplot(x='rat_arrival_number', y='food_availability', data=df, s=50, color='green')
        plt.xlabel("Rat Arrivals")
        plt.ylabel("Food Availability")
        plt.title("Food Availability vs Rat Arrivals")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
