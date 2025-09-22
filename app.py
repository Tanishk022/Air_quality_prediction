import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("regression_model.pkl", "rb"))

st.title("Air Quality Prediction App")

# User inputs
T = st.number_input("Enter Temperature (°C)", min_value=0.0, step=0.1)
TM = st.number_input("Enter Max Temp (°C)", min_value=0.0, step=0.1)
Tm = st.number_input("Enter Min Temp (°C)", min_value=0.0, step=0.1)
SLP  = st.number_input("Enter Sea Level Pressure", min_value=0.0, step=0.1)
H = st.number_input("Enter Humidity (%)", min_value=0.0, step=0.1)
VV = st.number_input("Enter Visibility (km)", min_value=0.0, step=0.1)
V = st.number_input("Enter Wind Direction", min_value=0.0, step=0.1)
VM  = st.number_input("Enter Wind Speed (km/h)", min_value=0.0, step=0.1)

if st.button("Predict Air Quality"):
    features = np.array([[T, TM, Tm, SLP, H, VV, V, VM]])
    prediction = model.predict(features)[0]

    st.subheader(f"Predicted AQI: {round(prediction,2)}")

    # Easy explanation block
    if prediction <= 50:
        st.success("✅ Good Air Quality – The air is clean and safe to go outside.")
    elif prediction <= 100:
        st.info("🙂 Moderate Air Quality – Acceptable air quality, but some pollutants may be present.")
    elif prediction <= 200:
        st.warning("⚠️ Unhealthy for Sensitive Groups – Children, elderly, and people with health issues should take precautions.")
    elif prediction <= 300:
        st.error("😷 Very Unhealthy – Everyone may experience health effects, avoid outdoor activities.")
    else:
        st.error("☠️ Hazardous – Serious health risks, stay indoors and avoid exposure.")




