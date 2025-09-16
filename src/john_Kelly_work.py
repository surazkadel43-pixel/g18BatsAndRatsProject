import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


bat_df = pd.read_csv('Data/dataset1.csv')

print(bat_df.columns)
 
sns.boxplot(data=bat_df, x='month', y='bat_landing_to_food')
plt.title('Bat Hesitation Time by Month')
plt.xlabel('Month')
plt.ylabel('Seconds from Landing to Food')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
 
df = pd.read_csv('Data/dataset1.csv', encoding='cp1252')
 
print("First 5 rows:")
print(df.head())
 
 
print("\nDescriptive statistics:")
print(df.describe())
print(f"Total rows used: {df.shape[0]}")
 
print("Average values for key columns:")
print("bat_landing_to_food:", df["bat_landing_to_food"].mean())
print("seconds_after_rat_arrival:", df["seconds_after_rat_arrival"].mean())
print("risk:", df["risk"].mean())
print("reward:", df["reward"].mean())
print("month:", df["month"].mean())
print("hours_after_sunset:", df["hours_after_sunset"].mean())
print("season:", df["season"].mean())
 
 
plt.figure(figsize=(8,5))
plt.hist(df["bat_landing_to_food"], bins=40, color='skyblue', edgecolor='black')
plt.title("Distribution of Bat Landing to Food Time")
plt.xlabel("Seconds")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
 
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='risk', y='bat_landing_to_food', palette='Set2')
plt.title("Feeding Time by Risk Behavior")
plt.xlabel("Risk")
plt.ylabel("Time from Landing to Food (s)")
plt.tight_layout()
plt.show()
 
 
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='reward', hue='risk', palette='coolwarm')
plt.title("Risk Behavior by Reward Outcome")
plt.xlabel("Reward (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(title='Risk')
plt.tight_layout()
plt.show()
 
 
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='season', y='hours_after_sunset', palette='Set3')
plt.title("Bat Activity Time by Season")
plt.xlabel("Season (0 = Dry, 1 = Wet)")
plt.ylabel("Hours After Sunset")
plt.tight_layout()
plt.show()
 
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='month', y='bat_landing_to_food', palette='pastel')
plt.title("Feeding Time by Month")
plt.xlabel("Month")
plt.ylabel("Seconds from Landing to Food")
plt.tight_layout()
plt.show()
 
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='season', hue='risk', palette='muted')
plt.title("Risk-Taking by Season")
plt.xlabel("Season")
plt.ylabel("Count")
plt.legend(title="Risk")
plt.tight_layout()
plt.show()