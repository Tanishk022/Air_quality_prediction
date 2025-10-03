import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("regression_model_2.pkl", "rb"))
# if st.button("Information about AQI Range"):
st.image("Screenshot 2025-10-03 113142.png")
col1,col2 = st.columns([2,2])
with col1:
    st.image("https://medicaldialogues.in/wp-content/uploads/2016/11/AIR-POLLUTION.jpg")
with col2:
    st.title("Air Quality Prediction App")

# st.image("Screenshot 2025-10-03 113142.png")
with st.sidebar:
# User inputs
    # City = st.text_input("City")fil
    PM10  = st.number_input("Enter PM10", min_value=0.0,max_value=50.0, step=1.0)
    PM2_5 = st.number_input("Enter PM2.5", min_value=0.0,max_value=30.0, step=1.0)
    NO2   = st.number_input("Enter NO2", min_value=0.0,max_value=40.0, step=1.0)
    O3    = st.number_input("Enter O3", min_value=0.0,max_value=50.0, step=1.0)
    CO    = st.number_input("Enter CO", min_value=0.0,max_value=1.0, step=1.0)
    SO2   = st.number_input("Enter SO2", min_value=0.0,max_value=40.0, step=1.0)
    NH3   = st.number_input("Enter NH3", min_value=0.0,max_value=200.0, step=1.0)
    

if st.button("Predict Air Quality"):

    features = np.array([[PM10,PM2_5, NO2, O3, CO, SO2,NH3]])


    # features = np.array([[T, TM, TT, SLP, H, VV, V, VM]])
    y_pred = model.predict(features)[0]
    # prediction = np.clip(prediction, 0, 500)


    st.subheader(f"Predicted AQI: {round(y_pred,2)}")

    # Easy explanation block
    st.subheader("Leval of Concern")
    if y_pred >= 0 and y_pred <= 51:
        st.subheader("Good Air Quality")
        st.success("There are negligible levels of pollutants like particulate matter and ground-level ozone, the air is fresh and crisp, and you can breathe easily without any respiratory discomfort.")

    elif y_pred >= 51 and y_pred <= 100:
        st.subheader("Moderate Air Quality")
        st.info("Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution..")
        
    elif y_pred >= 101 and y_pred <= 150:
        st.subheader("Unhealthy for Sensitive Groups")
        st.warning("Members of sensitive groups may experience health effects. The general public is less likely to be affected.")
    elif y_pred >= 151 and y_pred <= 200:
        st.subheader("Unhealthy")
        st.warning("Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.")
    elif y_pred >= 201 and y_pred <= 300:
        st.subheader("Very Unhealthy")
        st.warning("Health alert –The risk of health effects is increased for everyone.")
    else:
        st.write("Hazardous")
        st.error("– Serious health risks, stay indoors and avoid exposure.]")








