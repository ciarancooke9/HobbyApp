import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import joblib
from src.features.build_features import build_features

# --------------------
# Load model
# --------------------
@st.cache_resource
def load_model():
    return joblib.load("models/full_pipeline.joblib")

model = load_model()

# --------------------
# Title
# --------------------
st.title("🚕 Taxi Price Predictor")
st.write("Predict taxi trip prices based on trip details")

# --------------------
# Inputs
# --------------------
col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance (km)", 0.0, 200.0, 10.0)
    duration = st.number_input("Duration (minutes)", 0.0, 300.0, 25.0)
    passengers = st.slider("Passengers", 1, 6, 2)

with col2:
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    day_of_week = st.selectbox("Day Type", ["Weekday", "Weekend"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"])

# Pricing inputs
st.subheader("Pricing Parameters")

base_fare = st.number_input("Base Fare", 0.0, 10.0, 3.5)
per_km_rate = st.number_input("Per KM Rate", 0.0, 5.0, 1.0)
per_min_rate = st.number_input("Per Minute Rate", 0.0, 1.0, 0.3)

# --------------------
# Predict button
# --------------------
if st.button("💰 Predict Price"):

    # Create input dataframe
    input_data = pd.DataFrame([{
        "Trip_Distance_km": distance,
        "Time_of_Day": time_of_day,
        "Day_of_Week": day_of_week,
        "Passenger_Count": passengers,
        "Traffic_Conditions": traffic,
        "Weather": weather,
        "Base_Fare": base_fare,
        "Per_Km_Rate": per_km_rate,
        "Per_Minute_Rate": per_min_rate,
        "Trip_Duration_Minutes": duration
    }])
    input_data = build_features(input_data)
    # Predict
    prediction = model.predict(input_data)[0]

    # Output
    st.success(f"Predicted Price: €{prediction:.2f}")