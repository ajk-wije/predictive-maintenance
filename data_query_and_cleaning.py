import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/angelica.ginige/Desktop/Script/predictive-maintenance/predictive_maintenance.csv")

# Display basic info
print(df.info())

# Show first few rows
df.head()


#Check for Missing Values
df.isnull().sum()

# Select only numeric columns for median replacement
numeric_cols = df.select_dtypes(include=['number']).columns

# Fill missing values only for numeric columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())



#Summary Statistics
df.describe()


#Check Unique Values in Each Column
df.nunique()

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")

# Plot histograms of sensor values
df.hist(figsize=(12, 8), bins=30)
plt.show()


sns.countplot(x=df["Failure Type"])
plt.title("Distribution of Failure Types")
plt.xticks(rotation=45)
plt.show()

#Since sensor data changes over time, moving averages help capture trends

window_size = 5  # Adjust based on dataset frequency

for col in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()

#Lag features help the model understand past sensor readings
lag_periods = [1, 3, 5]  # Previous time steps to consider

for col in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
    for lag in lag_periods:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

#Multiply two features that might have relationships
df['torque_speed_interaction'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

#Convert failure types into a binary label for predictive maintenance
df['failure_label'] = df['Failure Type'].apply(lambda x: 1 if x != 'No Failure' else 0)

#Scaling numeric features improves model performance
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
sensor_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


#Save the processed dataset
df.to_csv("C:/Users/angelica.ginige/Desktop/Script/predictive-maintenance/predictive_maintenance_cleaned.csv", index=False)
