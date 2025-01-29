import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow API access from any frontend

# Dynamically get the model path
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")

# Check if model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load trained model
with open(model_path, "rb") as model_file:
    best_model = pickle.load(model_file)

@app.route('/')
def home():
    return "✅ Predictive Maintenance API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data
        data = request.json
        df = pd.DataFrame(data)

        # Define expected feature order
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

        # Check for missing features
        missing_features = [feature for feature in feature_order if feature not in df.columns]
        if missing_features:
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
    port = int(os.environ.get("PORT", 10000))  # Use port 10000 or environment port
    app.run(host='0.0.0.0', port=port, debug=True)
