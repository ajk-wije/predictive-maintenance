import streamlit as st
import pandas as pd
import requests

# Set API URL (make sure Flask API is running)
API_URL = "http://127.0.0.1:5000/predict"

# Streamlit UI
st.title("🔧 Predictive Maintenance Web App")
st.markdown("Enter sensor readings to predict machine failure.")

# Input fields for sensor values
air_temp = st.number_input("Air temperature [K]", min_value=250, max_value=400, value=300)
process_temp = st.number_input("Process temperature [K]", min_value=250, max_value=400, value=310)
rot_speed = st.number_input("Rotational speed [rpm]", min_value=500, max_value=5000, value=1500)
torque = st.number_input("Torque [Nm]", min_value=0, max_value=200, value=45)
tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=100, value=10)

# Derived feature
torque_speed_interaction = rot_speed * torque

# Button to make prediction
if st.button("🔍 Predict Machine Failure"):
    # Create input dictionary
    input_data = {
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rot_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "torque_speed_interaction": [torque_speed_interaction]
    }

    # Send request to Flask API
    response = requests.post(API_URL, json=input_data)
    
    # Display the prediction result
    if response.status_code == 200:
        result = response.json()
        prediction = result.get("Prediction", "Error")
        prob = result.get("Failure Probability", "N/A")
        
        if prediction == 1:
            st.error(f"⚠️ Machine Failure Predicted! (Failure Probability: {prob})")
        else:
            st.success(f"✅ No Failure Predicted. (Failure Probability: {prob})")
    else:
        st.error("❌ Error connecting to API. Make sure Flask is running.")

