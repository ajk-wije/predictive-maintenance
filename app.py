import os
import pickle
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow API access from any frontend

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")

def init_db():
    """Initialize SQLite database and create table if it doesn’t exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                air_temp REAL, process_temp REAL, rot_speed REAL, 
                torque REAL, tool_wear REAL, target INTEGER,
                air_temp_mean REAL, air_temp_std REAL,
                process_temp_mean REAL, process_temp_std REAL,
                rot_speed_mean REAL, rot_speed_std REAL,
                torque_mean REAL, torque_std REAL,
                tool_wear_mean REAL, tool_wear_std REAL,
                air_temp_lag_1 REAL, air_temp_lag_3 REAL, air_temp_lag_5 REAL,
                process_temp_lag_1 REAL, process_temp_lag_3 REAL, process_temp_lag_5 REAL,
                rot_speed_lag_1 REAL, rot_speed_lag_3 REAL, rot_speed_lag_5 REAL,
                torque_lag_1 REAL, torque_lag_3 REAL, torque_lag_5 REAL,
                tool_wear_lag_1 REAL, tool_wear_lag_3 REAL, tool_wear_lag_5 REAL,
                interaction REAL,
                prediction INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

init_db()  # Run database setup

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
        prediction = best_model.predict(df)[0]

        # Store prediction in database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    air_temp, process_temp, rot_speed, torque, tool_wear, target,
                    air_temp_mean, air_temp_std,
                    process_temp_mean, process_temp_std,
                    rot_speed_mean, rot_speed_std,
                    torque_mean, torque_std,
                    tool_wear_mean, tool_wear_std,
                    air_temp_lag_1, air_temp_lag_3, air_temp_lag_5,
                    process_temp_lag_1, process_temp_lag_3, process_temp_lag_5,
                    rot_speed_lag_1, rot_speed_lag_3, rot_speed_lag_5,
                    torque_lag_1, torque_lag_3, torque_lag_5,
                    tool_wear_lag_1, tool_wear_lag_3, tool_wear_lag_5,
                    interaction, prediction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                df.iloc[0, 0], df.iloc[0, 1], df.iloc[0, 2], df.iloc[0, 3], df.iloc[0, 4], df.iloc[0, 5],  # Main features
                df.iloc[0, 6], df.iloc[0, 7], df.iloc[0, 8], df.iloc[0, 9], df.iloc[0, 10], df.iloc[0, 11],  # Rolling means & std
                df.iloc[0, 12], df.iloc[0, 13], df.iloc[0, 14], df.iloc[0, 15],  # More rolling means & std
                df.iloc[0, 16], df.iloc[0, 17], df.iloc[0, 18], df.iloc[0, 19], df.iloc[0, 20], df.iloc[0, 21],  # Lags
                df.iloc[0, 22], df.iloc[0, 23], df.iloc[0, 24], df.iloc[0, 25], df.iloc[0, 26], df.iloc[0, 27],  # More Lags
                df.iloc[0, 28], df.iloc[0, 29], df.iloc[0, 30], df.iloc[0, 31],  # Final Lags + Interaction
                prediction
            ))
            conn.commit()

        return jsonify({"Prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/history', methods=['GET'])
def get_prediction_history():
    """Fetch last 10 predictions from the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10")
            records = cursor.fetchall()

        history = [
            {
                "air_temp": row[1],
                "process_temp": row[2],
                "rot_speed": row[3],
                "torque": row[4],
                "tool_wear": row[5],
                "interaction": row[31],
                "prediction": row[32],
                "timestamp": row[33]
            }
            for row in records
        ]

        return jsonify(history)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render’s dynamic port
    app.run(host='0.0.0.0', port=port, debug=True)
