import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Get the correct model path dynamically
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")

# Load the trained model
with open(model_path, "rb") as model_file:
    best_model = pickle.load(model_file)

# Define Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Predictive Maintenance API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json
        df = pd.DataFrame(data)

        # Ensure correct feature order
        feature_order = [
            'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
            'Torque [Nm]', 'Tool wear [min]', 'Target',  
            'Air temperature [K]_rolling_mean', 'Air temperature [K]_rolling_std',
            'Process temperature [K]_rolling_mean', 'Process temperature [K]_rolling_std',
            'Rotational speed [rpm]_rolling_mean', 'Rotational speed [rpm]_rolling_std',
            'Torque [Nm]_rolling_mean', 'Torque [Nm]_rolling_std',
            'Tool wear [min]_rolling_mean', 'Tool wear [min]_rolling_std',
            'Air temperature [K]_lag_1', 'Air temperature [K]_lag_3', 'Air temperature [K]_lag_5',
            'Process temperature [K]_lag_1', 'Process temperature [K]_lag_3', 'Process temperature [K]_lag_5',
            'Rotational speed [rpm]_lag_1', 'Rotational speed [rpm]_lag_3', 'Rotational speed [rpm]_lag_5',
            'Torque [Nm]_lag_1', 'Torque [Nm]_lag_3', 'Torque [Nm]_lag_5',
            'Tool wear [min]_lag_1', 'Tool wear [min]_lag_3', 'Tool wear [min]_lag_5',
            'torque_speed_interaction'
        ]

        # Ensure data contains all required features
        if not all(feature in df.columns for feature in feature_order):
            missing_features = list(set(feature_order) - set(df.columns))
            return jsonify({"error": f"Missing features in input: {missing_features}"})

        # Reorder columns to match model training
        df = df[feature_order]

        # Make prediction
        prediction = best_model.predict(df)

        # Return prediction result
        return jsonify({"Prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
