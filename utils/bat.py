import pandas as pd
class BatData:
    bat_data: pd.DataFrame

    def __init__(self, month: int = None):
        self.bat_data = pd.read_csv('Data/dataset1.csv')

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
    
    def divide_data_30_min_intervals(self):
        starting_min = 43
        min_interval = 30
        start_time = pd.to_datetime('26/12/2017 20:43',format="%d/%m/%Y %H:%M", errors="coerce")

        # every time hours_after_sunset decreases than thr upcoming value of hours_after_sunset increase last digit of starting_min by 1
        hours_after_sunset = self.bat_data[self.bat_data['start_time'] == start_time.strftime("%d/%m/%Y %H:%M")]['hours_after_sunset'].max()

        for index, row in self.bat_data.iterrows():
            if row['hours_after_sunset'] < hours_after_sunset:
                starting_min += 1
            hours_after_sunset = row['hours_after_sunset']
            
