import pandas as pd
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