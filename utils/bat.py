import sys, os

from matplotlib import pyplot as plt
from sklearn import metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from utils.rat import RatData
import statsmodels.stats.proportion as stm 
from utils.barchart import BarChart as BarChart
import scipy.stats as st 
import numpy as np
import statsmodels.api as sm

class BatData:
    bat_data: pd.DataFrame

    def __init__(self, month: int = None):
        self.bat_data = pd.read_csv('Data/bat_data_ordered.csv')

        if(month != None):
            self.bat_data = self.bat_data[self.bat_data['month'] == month]

        self.bat_data['timeFormat']  = pd.to_datetime(
                self.bat_data["start_time"], 
                format="%d/%m/%Y %H:%M",  
                    errors="coerce"
                )
        
        self.bat_data['hour'] = self.bat_data['timeFormat'].dt.hour
        #same for rat_period_start and rat_period_end nad sunset_time
        self.bat_data['rat_period_start']  = pd.to_datetime(
            self.bat_data["rat_period_start"], 
            format="%d/%m/%Y %H:%M",  
                errors="coerce"
            )
        self.bat_data['rat_period_end']  = pd.to_datetime(
            self.bat_data["rat_period_end"], 
            format="%d/%m/%Y %H:%M",  
                errors="coerce"
            )
        self.bat_data['sunset_time']  = pd.to_datetime(
            self.bat_data["sunset_time"], 
            format="%d/%m/%Y %H:%M",  
                errors="coerce"
            )


    def describeSelf(self):
        print(self.bat_data.describe())
    
    def order_data_by_time(self):
        self.bat_data = self.bat_data.sort_values(by=['timeFormat'])
        self.bat_data = self.bat_data.reset_index(drop=True)
        #export dataframe to csv but remove timeformat column
        self.bat_data = self.bat_data.drop(columns=['timeFormat'])
        self.bat_data.to_csv('Data/bat_data_ordered.csv', index=False)
    
    def add_time_format_column(self):
        self.bat_data['timeFormat']  = pd.to_datetime(
                self.bat_data["start_time"], 
                format="%d/%m/%Y %H:%M",  
                    errors="coerce"
                )
        self.bat_data['hour'] = self.bat_data['timeFormat'].dt.hour
        self.bat_data['minute'] = self.bat_data['timeFormat'].dt.minute
        self.bat_data['day'] = self.bat_data['timeFormat'].dt.day

    def merger_dataset_1_2_on_rat_period_start(self):
        rat_data = RatData()
        print(self.bat_data['rat_period_start'].iloc[1])
        # do this for every row in bat_data and make new dataframe with the results
        # Create separate columns for rat info
        self.bat_data[['rat_arrival_number', 'rat_minutes', 'bat_landing_number', 'time', 'food_availability']] = \
            self.bat_data.apply(lambda row: rat_data.get_data_by_time(row['rat_period_start']).iloc[0], axis=1)

        # export in data/bat_data_merged.csv
        self.bat_data.to_csv('Data/bat_data_merged.csv', index=False)
        # BarChart.plot_rat_bat_interactions(self.bat_data)
            
      
    
    # ======================
    # 📊 STATISTICAL TESTS
    # ======================

    def descriptive_stats(self):
        """Mean, median, IQR for key bat behaviour columns."""
        desc = {}
        for col in ['risk','bat_landing_to_food','seconds_after_rat_arrival','reward']:
            series = self.bat_data[col].dropna()
            desc[col] = {
                'mean': series.mean(),
                'median': series.median(),
                'IQR': series.quantile(0.75) - series.quantile(0.25)
            }
        return pd.DataFrame(desc)

    
    def risk_reward_distribution(self):
        """Count bats’ strategies based on risk vs reward."""
        counts = self.bat_data.groupby(['risk', 'reward']).size().unstack(fill_value=0)
        print("Risk–Reward distribution (rows=risk, cols=reward):")
        print(counts)
        return counts

    def risk_reward_confidence_interval(self):
        """Calculate z-based CI for easy vs hard way (risk–reward)."""
        
        # counts
        easy = len(self.bat_data[(self.bat_data['risk']==0) & (self.bat_data['reward']==1)])
        hard = len(self.bat_data[(self.bat_data['risk']==1) & (self.bat_data['reward']==1)])
        total = easy + hard

        results = {}
        for label, count in {"easy_way": easy, "hard_way": hard}.items():
            
            # confidence level
            conf_lvl = 0.95

            # proportion
            prop = count
            
            sig_lvl = 1 - conf_lvl
            
            
            ci_low, ci_upp = stm.proportion_confint(prop, total, alpha=sig_lvl, method="normal")

            results[label] = {
                "count": count,
                "proportion": round(prop/total, 3),
                "confidence_interval": (round(ci_low, 3), round(ci_upp, 3))
            }

        BarChart.plot_risk_reward_chart(results)
        return results
        
    def analyze_risk_vs_landing(self):
        """
        Process the data and perform hypothesis testing:
        Compare early vs late bat landings in terms of risk-taking.
        
        Returns a dictionary with:
        - median cutoff
        - early mean risk-taking
        - late mean risk-taking
        - t-statistic
        - p-value
        """
        df = self.bat_data.copy()
        
        # Median cutoff
        median_cutoff = df["seconds_after_rat_arrival"].median()
        
        # Define early vs late based on median
        df["timing"] = np.where(df["seconds_after_rat_arrival"] <= median_cutoff, "early", "late")
        
        # Split the risk data
        early_risk = df[df["timing"] == "early"]["risk"]
        late_risk = df[df["timing"] == "late"]["risk"]
        
        # Descriptive stats
        early_mean_risk = early_risk.mean()
        late_mean_risk = late_risk.mean()
        
        # One-sided t-test (early < late)
        t_stat, p_val = st.ttest_ind(early_risk, late_risk, alternative="less")
        
        BarChart.plot_early_late_risk(self.bat_data)
        BarChart.plot_seconds_after_rat(self.bat_data)
        return {
            "median_cutoff": median_cutoff,
            "early_mean_risk": early_mean_risk,
            "late_mean_risk": late_mean_risk,
            "t_stat": t_stat,
            "p_val": p_val
        }

    # apply mulitple linear regression modal to predict food availability based on hours after sunset and season
    def applyRegressionModel(self):
        df = self.bat_data.copy()
        
        df['actual_hours_after_sunset'] = (df['hours_after_sunset'] - (df['seconds_after_rat_arrival']/3600)).round(7) #until 7 decimal places
        
        coeff_season_0 = {'const': 4.0759, 'hours_after_sunset': -0.3169, 'season': 0} # this coeff is from dataset 2 regression modal when rat presence is 1 and season is 0
        coeff_season_1 = {'const': 0, 'hours_after_sunset': -0.1984, 'season': 3.5062} # this coeff is from dataset 2 regression modal when rat presence is 1 and season is 1
        # when season is 0 use coeff_season_0 else use coeff_season_1
        df['predicted_food_availability'] = df.apply(lambda row: coeff_season_0['const'] + coeff_season_0['hours_after_sunset'] * row['actual_hours_after_sunset'] +
                                                     coeff_season_0['season'] * row['season'] if row['season'] == 0 else coeff_season_1['const'] +
                                                     coeff_season_1['hours_after_sunset'] * row['actual_hours_after_sunset'] + coeff_season_1['season'] *
                                                     row['season'], axis=1)
        
                
        

        # df['predicted_food_availability'] = (coeff_season_0['const'] +
        #                                     coeff_season_0['hours_after_sunset'] * df['actual_hours_after_sunset'] +
        #                                     coeff_season_0['season'] * df['season']).round(7) #until 7 decimal places
    
        
        summary = {
            'mean_actual': df['food_availability'].mean(),
            'mean_predicted': df['predicted_food_availability'].mean(),
            'mse': metrics.mean_squared_error(df['food_availability'], df['predicted_food_availability']),
            'rmse': np.sqrt(metrics.mean_squared_error(df['food_availability'], df['predicted_food_availability'])),
            'mae': metrics.mean_absolute_error(df['food_availability'], df['predicted_food_availability']),
            'r2': metrics.r2_score(df['food_availability'], df['predicted_food_availability'])
        }
        updated_df = pd.read_csv('Data/bat_data_merged.csv')
        updated_df['predicted_food_availability'] = df['predicted_food_availability']
        updated_df['actual_hours_after_sunset'] = df['actual_hours_after_sunset']
        updated_df.to_csv('Data/bat_data_merged.csv', index=False)
        
        return summary

    def showScatterPlot(self,  season = 2,dataframe: pd.DataFrame = None, y_col = 'seconds_after_rat_arrival', x_col = [ 'predicted_food_availability', 'rat_minutes', 'season', 'actual_hours_after_sunset']):
        if dataframe is None:
            df = pd.read_csv('Data/bat_data_merged.csv')
        else:
            df = dataframe.copy()

        if season != 2:
            df = df[df['season'] == season]
        
        
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

        ax4.scatter(x = df[x_col[3]], y = df[y_col])
        ax4.set_xlabel(x_col[3])
        ax4.set_ylabel(y_col)

        plt.show()
        
    def multiple_linear_regression_of_bat_landing(self, season=2):
        """
        Model: bat_landing_to_food ~ actual_hours_after_sunset + season + food_availability + rat_minutes + seconds_after_rat_arrival
        """
        df = pd.read_csv('Data/bat_data_merged.csv')

        # Optional filtering
        if season != 2:
            df = df[df["season"] == season]
       

        

        X = df[["predicted_food_availability", "season", "rat_minutes", ]]
        #X = df[["rat_minutes",'bat_landing_number']]
        # y = df["seconds_after_rat_arrival"]
        y = df["bat_landing_to_food"]
        # Build regression
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        print(model.summary())
        return model
    
    def updated_bat_data(self):
        self.bat_data = pd.read_csv('Data/bat_data_merged.csv')
        return self.bat_data
