import requests
import json

# Define API endpoint
url = "http://127.0.0.1:5000/predict"

# Sample input (make sure it contains all expected features)
data = {
    "Air temperature [K]": [300],
    "Process temperature [K]": [310],
    "Rotational speed [rpm]": [1500],
    "Torque [Nm]": [45],
    "Tool wear [min]": [10],
    "Target": [0],

    # Rolling Mean & Std
    "Air temperature [K]_rolling_mean": [295], 
    "Air temperature [K]_rolling_std": [5],
    "Process temperature [K]_rolling_mean": [309], 
    "Process temperature [K]_rolling_std": [2],
    "Rotational speed [rpm]_rolling_mean": [1450], 
    "Rotational speed [rpm]_rolling_std": [50],
    "Torque [Nm]_rolling_mean": [40], 
    "Torque [Nm]_rolling_std": [3],
    "Tool wear [min]_rolling_mean": [9], 
    "Tool wear [min]_rolling_std": [1],

    # Lag Features
    "Air temperature [K]_lag_1": [298],
    "Air temperature [K]_lag_3": [299],
    "Air temperature [K]_lag_5": [300],
    "Process temperature [K]_lag_1": [308],
    "Process temperature [K]_lag_3": [309],
    "Process temperature [K]_lag_5": [310],
    "Rotational speed [rpm]_lag_1": [1480],
    "Rotational speed [rpm]_lag_3": [1490],
    "Rotational speed [rpm]_lag_5": [1500],
    "Torque [Nm]_lag_1": [42],
    "Torque [Nm]_lag_3": [43],
    "Torque [Nm]_lag_5": [44],
    "Tool wear [min]_lag_1": [8],
    "Tool wear [min]_lag_3": [9],
    "Tool wear [min]_lag_5": [10],

    # Interaction Feature
    "torque_speed_interaction": [1500 * 45]
}

# Send POST request
response = requests.post(url, json=data)

# Print API response
print("Response Status Code:", response.status_code)
print("API Response:", response.json())
