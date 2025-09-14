import pandas as pd
import numpy as np
from scipy import stats

import statsmodels.api as sm

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
     """Correlation between rat activity and food availability."""
     corr = self.rat_data[['rat_minutes', 'food_availability']].corr()
     return corr.applymap(float).round(3)  # clean + rounded

    def regression_food_on_rats(self):
     """Regression: predict food availability from rat arrivals."""
     X = self.rat_data[['rat_arrival_number']]
     y = self.rat_data['food_availability']
     X = sm.add_constant(X)  # add intercept
    
     model = sm.OLS(y, X).fit()
    
     return {
        "intercept": float(model.params['const']),
        "slope_rat_arrival": float(model.params['rat_arrival_number']),
        "r_squared": float(model.rsquared),
        "p_value": float(model.pvalues['rat_arrival_number']),
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
