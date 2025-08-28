# Cleaning data and Data wrangling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_hourly_frequency(freq_series, title="Event Frequency per Hour"):
    plt.style.use('fivethirtyeight')
    """
    Plots frequency of events across 24 hours.
    
    Parameters:
        freq_series (pd.Series): Index = hour (0–23), Values = frequency counts
        title (str): Title of the plot
    """
    plt.figure(figsize=(12,6))
    plt.bar(freq_series.index, freq_series.values, color="skyblue", edgecolor="black")
    plt.xticks(range(24))  # show all 24 hours on x-axis
    plt.xlabel("Hour of Day")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
    
def get24HourTimelineForMonth(df: pd.DataFrame, month: int) -> pd.Series:
    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Ensure start_time is datetime
    df['start_time'] = pd.to_datetime(df['start_time'], format="%d/%m/%Y %H:%M", errors='coerce')

    # Filter by the month column
    monthly_data = df[df['month'] == month].copy()

    # Create 'hour' column
    monthly_data.loc[:, 'hour'] = monthly_data['start_time'].dt.hour

    # Count events per hour and fill missing hours with 0
    return monthly_data.groupby('hour').size().reindex(range(24), fill_value=0)

def removeOutliers(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """
    Removes outliers from a DataFrame based on a z-score threshold.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to check for outliers.
        threshold (float): The z-score threshold above which to consider values as outliers.

    Returns:
        pd.DataFrame: A DataFrame with outliers removed.
    """
    # Calculate z-scores
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    # Filter out outliers
    return df[abs(z_scores) < threshold]

def plotHistogram(df: pd.DataFrame, column: str, title: str, xlabel: str, ylabel: str):
    """
    Plots a histogram of a specified column in a DataFrame. And also add [25,50, mean, 75,80,90] percentiles.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=30, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add percentile lines
    percentiles = [25, 50, 75, 80, 90]
    for p in percentiles:
        plt.axvline(df[column].quantile(p / 100), color='blue', linestyle='dashed', linewidth=1)
        plt.text(df[column].quantile(p / 100), 5, f'{p}%', color='red', ha='center')

    plt.show()


BatDf = pd.read_csv('dataset1.csv')
RatDf = pd.read_csv('dataset2.csv')


#Bat DataFrame for April
batDfApril = BatDf[BatDf["month"] == 4].copy()
# batDfApril = BatDf.copy()
batDfApril['timeFormat']  = pd.to_datetime(
    batDfApril["start_time"], 
    format="%d/%m/%Y %H:%M",  
    errors="coerce"
)
batDfApril['hour'] = batDfApril['timeFormat'].dt.hour



ratDfApril = RatDf[RatDf["month"] == 4].copy()
# ratDfApril = RatDf.copy()
ratDfApril['timeFormat']  = pd.to_datetime(
    ratDfApril["time"], 
    format="%d/%m/%Y %H:%M", 
    errors="coerce"
)

#Create an hour column and assign in 24hr format
ratDfApril['hour'] = ratDfApril['timeFormat'].dt.hour


def plotBarChart(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str):
    """
    Plots a bar chart of a specified column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x (str): The column name to use for the x-axis.
        y (str): The column name to use for the y-axis.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(df[x], df[y], color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xticks(range(24))  # show all 24 hours on x-axis
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def addColumnValuePerHour(df: pd.DataFrame, groupby_col: str, sum_col: str, new_col: str, mean = False) -> pd.DataFrame:
    """
    Adds a new column to the DataFrame with the sum of a specified column, grouped by another column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        groupby_col (str): The column name to group by.
        sum_col (str): The column name to sum.
        new_col (str): The new column name to create.
    """
    summary = (
        df.groupby(groupby_col)
        .agg(**{new_col: (sum_col, 'sum')})
    )
    if mean:
        summary = (
            df.groupby(groupby_col)
            .agg(**{new_col: (sum_col, 'mean')})
        )
        
    return summary.reindex(range(24), fill_value=0).reset_index()

new_Rat_Df_April = pd.DataFrame()


new_Rat_Df_April['hour'] = range(24)
new_Rat_Df_April['rat_count'] = addColumnValuePerHour(ratDfApril, 'hour', 'rat_arrival_number', 'rat_count')[['rat_count']]
new_Rat_Df_April['bat_count'] = addColumnValuePerHour(ratDfApril, 'hour', 'bat_landing_number', 'bat_count')[['bat_count']]
new_Rat_Df_April['food_available'] = addColumnValuePerHour(ratDfApril, 'hour', 'food_availability', 'food_available')[['food_available']]
new_Rat_Df_April['rat_min_average'] = addColumnValuePerHour(ratDfApril, 'hour', 'rat_minutes', 'rat_min_average', mean=True)[['rat_min_average']]
new_Rat_Df_April['rat_min_Total'] = addColumnValuePerHour(ratDfApril, 'hour', 'rat_minutes', 'rat_min_Total')[['rat_min_Total']]


# plotBarChart(new_Rat_Df_April, 'hour', 'rat_count', 'Rat Arrival Count per Hour in April', 'Hour of Day', 'Rat Arrival Count')
# plotBarChart(new_Rat_Df_April, 'hour', 'bat_count', 'Bat Landing Count per Hour in April', 'Hour of Day', 'Bat Landing Count')
# plotBarChart(new_Rat_Df_April, 'hour', 'food_available', 'Food Availability per Hour in April', 'Hour of Day', 'Food Availability')
# plotBarChart(new_Rat_Df_April, 'hour', 'rat_min_average', 'Average Rat Minutes per Hour in April', 'Hour of Day', 'Average Rat Minutes')
# plotBarChart(new_Rat_Df_April, 'hour', 'rat_min_Total', 'Total Rat Minutes per Hour in April', 'Hour of Day', 'Total Rat Minutes')


###################################################################################################

def calculateRiskAndRewardPerHour(batDfApril: pd.DataFrame) -> pd.DataFrame:
    # Count occurrences grouped by (risk, reward, hour)
    grouped = (
        batDfApril.groupby(['risk', 'reward', 'hour'])
        .size()
        .reset_index(name="count")
    )
    


    # Pivot so each (risk, reward) combination gets its own column
    pivoted = grouped.pivot_table(
        index="hour",
        columns=["risk", "reward"],
        values="count",
        fill_value=0
    )
    
    

    # Flatten column names: (risk, reward) → "R{risk}_RE{reward}"
    pivoted.columns = [f"R{r}_RE{re}" for r, re in pivoted.columns]

    # Ensure all 24 hours are present
    pivoted = pivoted.reindex(range(24), fill_value=0)
    
 

  # Convert values to integers (no decimals)
    pivoted = pivoted.astype(int)
    
    return pivoted.reset_index()


time_difference = (
    pd.to_datetime(batDfApril["rat_period_end"], format="%d/%m/%Y %H:%M", errors="coerce")
    - pd.to_datetime(batDfApril["rat_period_start"], format="%d/%m/%Y %H:%M", errors="coerce")
)

batDfApril['rat_period_end'] = pd.to_datetime(batDfApril["rat_period_end"], format="%d/%m/%Y %H:%M", errors="coerce")
batDfApril['rat_period_start'] = pd.to_datetime(batDfApril["rat_period_start"], format="%d/%m/%Y %H:%M", errors="coerce")
# Use .dt.total_seconds() to calculate the difference in seconds for each row
batDfApril['rat_period_seconds'] = time_difference.dt.total_seconds() / 60


new_Bat_Df_April = pd.DataFrame()

new_Bat_Df_April['hour'] = range(24)
new_Bat_Df_April['rat_period_minutes'] = addColumnValuePerHour(batDfApril, 'hour', 'rat_period_seconds', 'rat_period_minutes', mean=True)[['rat_period_minutes']]
new_Bat_Df_April['bat_seconds_after_rat_arrival'] = addColumnValuePerHour(batDfApril, 'hour', 'seconds_after_rat_arrival', 'bat_seconds_after_rat_arrival', mean=True)[['bat_seconds_after_rat_arrival']]
new_Bat_Df_April['bat_landing_to_food_after_Landing'] = addColumnValuePerHour(batDfApril, 'hour', 'bat_landing_to_food', 'bat_landing_to_food_after_Landing', mean=True)[['bat_landing_to_food_after_Landing']]
new_Bat_Df_April['RI_0_RE_0'] = calculateRiskAndRewardPerHour(batDfApril)['R0_RE0']
new_Bat_Df_April['RI_0_RE_1'] = calculateRiskAndRewardPerHour(batDfApril)['R0_RE1']
new_Bat_Df_April['RI_1_RE_0'] = calculateRiskAndRewardPerHour(batDfApril)['R1_RE0']
new_Bat_Df_April['RI_1_RE_1'] = calculateRiskAndRewardPerHour(batDfApril)['R1_RE1']

print(batDfApril.groupby(['risk', 'reward']).size())


# plotBarChart(new_Bat_Df_April, 'hour', 'rat_period_minutes', 'Average Rat Period in min  in April', 'Hour of Day', 'Average Rat Period in min')
# plotBarChart(new_Bat_Df_April, 'hour', 'bat_seconds_after_rat_arrival', 'Average Bat Seconds After Rat Arrival per Hour in April', 'Hour of Day', 'Average Bat Seconds After Rat Arrival')
# plotBarChart(new_Bat_Df_April, 'hour', 'bat_landing_to_food_after_Landing', 'Average Bat Landing to Food After Landing in seconds per Hour in April', 'Hour of Day', 'Average Bat Landing to Food After Landing in seconds')
# show bar chart of bat_count
#plotBarChart(new_Rat_Df_April, 'hour', 'bat_count', 'Average Bat Count per Hour in April', 'Hour of Day', 'Average Bat Count')


#now find z index for how many bats goes for risk and reward



#Merging two DataFrames
new_rat_bat_df = pd.merge(new_Rat_Df_April, new_Bat_Df_April, on='hour', how='inner')
print(new_rat_bat_df.count())

# # Data
x = new_rat_bat_df['hour']
rat = new_rat_bat_df['rat_count']
food = new_rat_bat_df['food_available']
bat = new_rat_bat_df['bat_count']

# Set bar width
bar_width = 0.25  

# Create bar positions
r1 = np.arange(len(x))               # positions for rat
r2 = r1 + bar_width                  # shift for food
r3 = r1 + 2*bar_width                # shift for bat
r4 = r1 + 2*bar_width                # shift for RI_0_RE_0
r5 = r1 + 3*bar_width                # shift for RI_0_RE_1
r6 = r1 + 4*bar_width                # shift for RI_1_RE_0
r7 = r1 + 3*bar_width                # shift for RI_1_RE_1

# Plot bars
plt.figure(figsize=(14,6))
plt.bar(r1, rat, color='blue', width=bar_width, label='Rat Count')
plt.bar(r2, food, color='green', width=bar_width, label='Food Availability')
# plt.bar(r3, bat, color='red', width=bar_width, label='Bat Count')
plt.bar(r4, new_rat_bat_df['RI_1_RE_0'], color='yellow', width=bar_width, label='Risk 0 Reward 1')
# plt.bar(r4, new_rat_bat_df['RI_0_RE_0'], color='orange', width=bar_width, label='Risk 0 Reward 0')

# plt.bar(r6, new_rat_bat_df['RI_1_RE_0'], color='purple', width=bar_width, label='Risk 1 Reward 0')
# plt.bar(r7, new_rat_bat_df['RI_1_RE_1'], color='pink', width=bar_width, label='Risk 1 Reward 1')

# X-axis labels (hours)
plt.xticks(r1 + bar_width, x)  

plt.xlabel("Hour")
plt.ylabel("Values")
plt.title("Rat Count vs Food Availability vs Bat Count per Hour")
plt.legend()
plt.show()