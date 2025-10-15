import sys, os

from sklearn.discriminant_analysis import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from math import ceil
import statsmodels.api as sm
from utils.barchart import BarChart as BarChart
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from typing import List, Optional, Tuple
class RatData:
    rat_data: pd.DataFrame
    num_of_month: int

    def __init__(self, month: int = None):
        self.rat_data = pd.read_csv('Data/dataset2.csv')

        if(month != None):
            self.rat_data = self.rat_data[self.rat_data['month'] == month]

        self.rat_data['timeFormat']  = pd.to_datetime(
                self.rat_data["time"], 
                format="%d/%m/%Y %H:%M",  
                    errors="coerce"
                )
        self.rat_data['hour'] = self.rat_data['timeFormat'].dt.hour
        self.num_of_month = self.rat_data['month'].nunique()
        #if  month is less than 3  add season column and make it 0 and if month is greater and equal to 3 make it 1
    
        self.rat_data['season'] = np.where(self.rat_data['month'] < 3, 0, 1)
        
        # Compute average rat minutes safely (avoid division by zero)
        self.rat_data['rat_avg_minutes'] = np.where(
            self.rat_data['rat_arrival_number'] > 0,
            self.rat_data['rat_minutes'] / self.rat_data['rat_arrival_number'],
            0
        )
        
        #rat presence column if rat is greater than 0 then 1 else 0
        self.rat_data['rat_present'] = (self.rat_data['rat_arrival_number'] > 0).astype(int)
        
       


    def describeSelf(self):
        print(self.rat_data.describe())

    def count_food_availability_per_hour(self):

        self.rat_data['avg_food_availability_hourly'] = self.rat_data.groupby('hour')['food_availability'].transform('mean') 
        
    
    def count_rat_arrivals_per_hour(self):
        self.rat_data['rat_arrivals'] = 1
        self.rat_data['avg_rat_arrivals_hourly'] = self.rat_data.groupby('hour')['rat_arrival_number'].transform('mean') 
    
    def descriptive_stats(self):
        """Return descriptive stats (mean, median, IQR) for key columns."""
        desc = {}
        for col in ['bat_landing_number', 'food_availability', 'rat_arrival_number']:
            series = self.rat_data[col].dropna()
            desc[col] = {
                'mean': series.mean(),
                'median': series.median(),
                'IQR': series.quantile(0.75) - series.quantile(0.25)
            }
        return pd.DataFrame(desc)

       
       
    def confidence_intervals(self, confidence=0.95):
     """Compute and print confidence intervals for key variables."""
     for col in ['bat_landing_number', 'food_availability', 'rat_arrival_number']:
        series = self.rat_data[col].dropna()
        mean = series.mean()
        sem = stats.sem(series)
        margin = sem * stats.t.ppf((1 + confidence) / 2., len(series)-1)
        low, high = mean - margin, mean + margin
        print(f"{col}: 95% CI = ({low:.2f}, {high:.2f})")


    def ttest_bat_landings(self):
        """
        Compare bat landings when rats are present vs absent (Welch's t-test).
        Uses raw data to compute t-statistic and p-value.
        """
        # Step 1: Separate groups
        with_rats = self.rat_data[self.rat_data['rat_arrival_number'] > 0]['bat_landing_number']
        without_rats = self.rat_data[self.rat_data['rat_arrival_number'] == 0]['bat_landing_number']

        # Step 2: Compute sample statistics
        mean1 = np.mean(with_rats)
        mean2 = np.mean(without_rats)
        std1 = np.std(with_rats, ddof=1)
        std2 = np.std(without_rats, ddof=1)
        n1 = len(with_rats)
        n2 = len(without_rats)
        
        #print all the above values
        print(f"With Rats: mean={mean1:.2f}, std={std1:.2f}, n={n1}")
        print(f"Without Rats: mean={mean2:.2f}, std={std2:.2f}, n={n2}")
        


        # Step 3: Use scipy to compute Welch's t-test from summary stats
        t_stat, p_value = stats.ttest_ind_from_stats(
            mean1=mean1, std1=std1, nobs1=n1,
            mean2=mean2, std2=std2, nobs2=n2,
            equal_var=False,          # Welch's t-test
            alternative='two-sided'   # two-tailed test
        )

        # Step 4: Return results
        return {
            "t-statistic": float(t_stat),
            "p-value": float(p_value)
        }
        
        
    def correlation_rat_food(self):
        """Step-by-step correlation between rat activity and food availability,
        handles zero-inflated data."""
        
        df = self.rat_data[['rat_arrival_number', 'food_availability']].copy()
        
        
        # Step 0: create presence/absence column
        df['rat_present'] = (df['rat_arrival_number'] > 0).astype(int)
        
        # Step 1: Means for each group
        means = df.groupby('rat_present')['food_availability'].mean()
        print("--- Step 1: Mean Food Availability by Presence ---")
        print(means)
        
        # Step 2: Visualize using boxplot via BarChart static method
        BarChart.plotPresenceBoxplot(df, zero_inflated_col='rat_arrival_number', numeric_col='food_availability')
        
        return means

    def correlation_rat_bats(self):
        """Step-by-step correlation between rat activity and bat landings,
        handles zero-inflated data."""

        df = self.rat_data[['rat_arrival_number', 'bat_landing_number']].copy()

        # Step 0: create presence/absence column
        df['rat_present'] = (df['rat_arrival_number'] > 0).astype(int)

        # Step 1: Means for each group
        means = df.groupby('rat_present')['bat_landing_number'].mean()
        print("--- Step 1: Mean Bat Landings by Presence ---")
        print(means)

        # Step 2: Visualize using boxplot via BarChart static method
        BarChart.plotPresenceBoxplot(df, zero_inflated_col='rat_arrival_number', numeric_col='bat_landing_number')

        return means

        # Step 0: create presence/absence column
        df['rat_present'] = (df['rat_minutes'] > 0).astype(int)
        
        # Step 1: Means for each group
        means = df.groupby('rat_present')['food_availability'].mean()
        print("--- Step 1: Mean Food Availability by Presence ---")
        print(means)
        
        # Step 2: Visualize using boxplot via BarChart static method
        BarChart.plotPresenceBoxplot(df, zero_inflated_col='rat_minutes', numeric_col='food_availability')
        
        return means
    def regression_food_on_rats(self):
     """Regression: predict food availability from rat arrivals."""
     X = self.rat_data[['rat_arrival_number']]
     y = self.rat_data['food_availability']
     X = sm.add_constant(X)  # add intercept
    
     model = sm.OLS(y, X).fit()
    
     regression_result =  {
        "intercept": float(model.params['const']),
        "slope_rat_arrival": float(model.params['rat_arrival_number']),
        "r_squared": float(model.rsquared),
        "p_value": float(model.pvalues['rat_arrival_number']),
        "f_statistic": float(model.fvalue),
        "n_observations": int(model.nobs)
     }
     BarChart.plot_regression(self.rat_data, 'rat_arrival_number', 'food_availability', regression_result)
     return regression_result
     
    def regression_bat_on_rats_food(self):
        """
        Multiple regression:
        Predict bat landing number from rat arrivals and food availability.
        Model: bat_landing_number ~ rat_arrival_number + food_availability
        """
        # Define X (predictors) and y (outcome)
        X = self.rat_data[['rat_arrival_number', 'food_availability']]
        y = self.rat_data['bat_landing_number']

        # Add constant for intercept
        X = sm.add_constant(X)

        # Fit OLS model
        model = sm.OLS(y, X).fit()

        # Return clean results
        return {
            "intercept": float(model.params['const']),
            "slope_rat_arrival": float(model.params['rat_arrival_number']),
            "slope_food_availability": float(model.params['food_availability']),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "p_values": {k: float(v) for k, v in model.pvalues.items()},
            "f_statistic": float(model.fvalue),
            "n_observations": int(model.nobs)
        }

    def get_data_by_time(self, timestamp: pd.Timestamp):
        """
        Returns rat and bat data for the 30-min interval containing the given timestamp.
        """
        # Convert input timestamp to datetime if it is a string
        if isinstance(timestamp, str):
            ts = pd.to_datetime(timestamp, format="%d/%m/%Y %H:%M", errors='coerce')
        else:
            ts = timestamp

        if ts is pd.NaT:
            raise ValueError("Invalid timestamp format")

        # Find the row where timestamp falls within the 30-min period
        row = self.rat_data[
            (ts >= self.rat_data['timeFormat']) &
            (ts < self.rat_data['timeFormat'] + pd.Timedelta(minutes=30))
        ]

        # Return only the desired columns
        return row[['rat_arrival_number', 'rat_minutes', 'bat_landing_number', 'time', 'food_availability']]


    def summarize_bat_food_by_rat_presence(self):
        """
        Summarize bat landing number and food availability 
        by rat presence (rat_arrival_number > 0 vs = 0).
        """
        # Create rat presence column
        self.rat_data['rat_present'] = (self.rat_data['rat_arrival_number'] > 0).astype(int)
        
        # Define aggregation functions
        agg_dict = {
            'bat_landing_number': ['mean', 'median', lambda x: x.quantile(0.75) - x.quantile(0.25)],  # IQR
            'food_availability': 'mean'
        }
        
        summary = self.rat_data.groupby('rat_present').agg(agg_dict)
        
        # Rename the IQR column
        summary.columns = ['bat_mean', 'bat_median', 'bat_IQR', 'food_mean']
        
        return summary
    
    def multipleRegressionModel(self, 
                             rat_presence=2, 
                             season=2, 
                             dataframe: pd.DataFrame=None, 
                             y_col='bat_landing_number', 
                             x_col=[  'hours_after_sunset','rat_minutes', 'season', 'food_availability']):
        """
        Builds a Multiple Linear Regression model to predict a target variable (y_col)
        based on selected explanatory variables (x_col), with optional filtering 
        by rat presence and season.

        Parameters:
            rat_presence (int): 0 = no rats, 1 = rats present, 2 = all data
            season (int): 0 = winter, 1 = spring, 2 = all seasons
            dataframe (pd.DataFrame): dataset to use (default = self.rat_data)
            y_col (str): dependent variable (target)
            x_col (list): explanatory variables

        Returns:
            model.summary(): statsmodels regression summary
        """

        # Step 1: Choose dataframe
        df = dataframe if dataframe is not None else self.rat_data.copy()

        # Step 2: Apply filters
        if season != 2:
            df = df[df['season'] == season]
        if rat_presence != 2:
            df = df[df['rat_present'] == rat_presence]

        # Step 3: Prepare X and y
        X = df[x_col].copy()
        y = df[y_col].copy()
        
        # Step 4: Remove missing values
        df_clean = pd.concat([X, y], axis=1).dropna()
        X = df_clean[x_col]
        y = df_clean[y_col]

        # Step 5: Add constant and fit model
        X = sm.add_constant(X)
        print(X.info())  # optional debugging info
        model = sm.OLS(y, X).fit()

        return model.summary()
    
    def multiple_linear_regression_with_interaction(self, rat_presence=2, season=2):
        """
        Model: food_availability ~ hours_after_sunset + season + (hours_after_sunset * season)
        """
        df = self.rat_data.copy()

        # Optional filtering
        if season != 2:
            df = df[df["season"] == season]
        if rat_presence != 2:
            df = df[df["rat_present"] == rat_presence]

        # Define variables
        df["interaction"] = df["hours_after_sunset"] * df["rat_minutes"]

        X = df[["hours_after_sunset", "season", "rat_minutes", "food_availability", "interaction"]]
        #X = df[["rat_minutes",'bat_landing_number']]
        # y = df["food_availability"]
        y = df["bat_landing_number"]
        # Build regression
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        print(model.summary())
        return model
    
    def multiple_linear_regression_bat_landing(self, rat_presence = 2, season = 2):
        df = self.rat_data.copy()
        if season != 2:
            df = df[df['season'] == season]
        if rat_presence != 2:
            df = df[df['rat_present'] == rat_presence]


        X = df[['rat_minutes','hours_after_sunset', 'season', 'food_availability']]
        y = df['bat_landing_number']
        print(X.info())
        X = sm.add_constant(X)  # add intercept
        model = sm.OLS(y, X).fit()
        return model.summary()
    
    #multiple linear regression modal reponse variable ( food_availability ) and 
    # explanatory variables (   hours_after_sunset, rat_avg_minutes, month, season) using LinearRegression from sklearn and also show graph of the regression
    def multiple_linear_regression_sklearn(self, rat_presence = 2, season = 2):
        df = self.rat_data.copy()
        if season != 2:
            df = df[df['season'] == season]
        
        if rat_presence != 2:
            df = df[df['rat_present'] == rat_presence]
        
        X = df[['rat_minutes','hours_after_sunset', 'season', 'food_availability']]
        y = df['bat_landing_number']
        #split the data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)
        print("dataframe shape:", df.shape)
        print("X_train size:", X_train.shape)
        print("X_test size:", X_test.shape)
        model = LinearRegression()
        model.fit(X_train, y_train)
        # Print the intercept and coefficient learned by the linear regression model
        print("Intercept:", model.intercept_)
        print("Coefficients:", model.coef_)
        # Predict on the test data
        y_pred = model.predict(X_test)
        # Print evaluation metrics
        print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
        print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
        print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        #r-squared also called coefficient of determination
        print("R^2:", metrics.r2_score(y_test, y_pred))
        # Plotting actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Food Availability')
        plt.show()
        
        return model

    def showScatterPlot(self, rat_presence = 2, season = 2,dataframe: pd.DataFrame = None, y_col = 'bat_landing_number', x_col = [ 'food_availability', 'rat_minutes', 'season', 'hours_after_sunset']):
        if dataframe is None:
            df = self.rat_data.copy()
        else:
            df = dataframe.copy()

        if season != 2:
            df = df[df['season'] == season]
        if rat_presence != 2:
            df = df[df['rat_present'] == rat_presence]
        
       
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        fig1.tight_layout()
        #add label season at the top inside graph
        fig1.suptitle(f'Scatter Plots for  Season={season}')
        
        
        """ FIGURE 1 """
        ax1.scatter(x = df[x_col[0]], y = df[y_col])
        ax1.set_xlabel(x_col[0])
        ax1.set_ylabel(y_col)

        ax2.scatter(x = df[x_col[1]], y = df[y_col])
        ax2.set_xlabel(x_col[1])
        ax2.set_ylabel(y_col)

        ax3.scatter(x = df[x_col[2]], y = df[y_col])
        ax3.set_xlabel(x_col[2])
        ax3.set_ylabel(y_col)
        
        if len(x_col) > 3:
            ax4.scatter(x = df[x_col[3]], y = df[y_col])
            ax4.set_xlabel(x_col[3])
            ax4.set_ylabel(y_col)

        plt.show()
    #linear regression modal between food avaibility and bat landing number
    
    #Generalization function for Applying Z- scale Standardization
    def build_and_evaluate_regression_using_z_stand_rescale(self, rat_presence = 2, season = 2, dataframe = None, y_col = 'bat_landing_number', x_col = [ 'food_availability', 'rat_minutes', 'season', 'hours_after_sunset']):
        """
        Build and evaluate a linear regression model using Statsmodels.
        Applies z-score standardisation and compares results before and after standardisation.


        Returns
        -------
        dict
            Dictionary containing model summaries before and after standardisation.
        """

        
        
        # Step 2: Separate explanatory (X) and response (y) variables
        if dataframe is None:
            df = self.rat_data.copy()
        else:
            df = dataframe.copy()

        if season != 2:
            df = df[df['season'] == season]
        if rat_presence != 2:
            df = df[df['rat_present'] == rat_presence]
            # X, y
        X = df[x_col]  # 2-D DataFrame
        y = df[y_col]           # 1-D Series
        
        # Step 3: Build and evaluate the original linear regression model
        X_const = sm.add_constant(X)
        model_original = sm.OLS(y, X_const).fit()
        print("\n--- Original Regression Model Summary ---")
        print(model_original.summary())

        self.showScatterPlot(rat_presence=rat_presence, season=season,  y_col=y_col, x_col=x_col)

        
        
    
    

    