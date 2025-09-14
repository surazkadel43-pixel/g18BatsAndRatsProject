import pandas as pd

from utils.rat import RatData

import statsmodels.stats.proportion as stm 


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

    def risk_reward_confidence_interval(self, confidence=0.95):
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

        return results
        

