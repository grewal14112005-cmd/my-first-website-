
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Garry's AI Disaster Response System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Garry's AI Disaster Response System")
st.caption("AI-assisted flood-risk classification prototype")

st.warning(
    "Educational prototype using synthetic training data. "
    "Not suitable for real-world flood warnings or emergency decisions."
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "flood_risk_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please add flood_risk_model.pkl to the repository.")
    st.stop()

model = load_model()

st.subheader("Enter environmental conditions")

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=85.0
)

river_level = st.number_input(
    "River level (m)",
    min_value=0.0,
    max_value=20.0,
    value=2.7
)

soil_saturation = st.slider(
    "Soil saturation (%)",
    min_value=0,
    max_value=100,
    value=72
)

if st.button("Predict flood-risk category"):
    new_data = pd.DataFrame([{
        "rainfall_mm": rainfall,
        "river_level_m": river_level,
        "soil_saturation_pct": soil_saturation
    }])

    prediction = model.predict(new_data)[0]

    st.subheader("Model result")
    st.metric("Predicted category", prediction)

    st.write("Input summary")
    st.dataframe(new_data, use_container_width=True)

st.divider()
st.subheader("About this project")
st.write(
    "Developed by Garry as a computer science engineering project "
    "exploring machine learning and emergency-response support."
)
