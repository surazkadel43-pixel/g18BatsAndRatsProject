import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load your data
df = pd.read_csv('dataset1.csv')

# Feature engineering example
df['season'] = df['season'].astype(int)  # ensure numeric
df['risk'] = df['risk'].astype(int)
df['reward'] = df['reward'].astype(int)

# Example: create rat_presence_duration if columns exist
if 'rat_period_end' in df.columns and 'rat_period_start' in df.columns:
    df['rat_presence_duration'] = pd.to_datetime(df['rat_period_end']) - pd.to_datetime(df['rat_period_start'])
    df['rat_presence_duration'] = df['rat_presence_duration'].dt.total_seconds().fillna(0)

# Select explanatory variables
X = df[['risk', 'reward', 'bat_landing_to_food', 'hours_after_sunset', 'season']]
if 'rat_presence_duration' in df.columns:
    X = X.join(df['rat_presence_duration'])

# Response variable
y = df['seconds_after_rat_arrival']

# Handle missing data if any
X = X.fillna(0)
y = y.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Optional: coefficients
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)








