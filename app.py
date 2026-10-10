

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Garry's AI Disaster Response System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Garry's AI Disaster Response System")
st.caption("Developed by Garry | AI/ML Engineering Project")

st.warning(
    "Educational prototype using synthetic data. "
    "This is not a real flood-warning system."
)

# Demonstration dataset
data = {
    "rainfall_mm": [10, 25, 40, 60, 80, 100, 120, 150, 35, 90],
    "river_level_m": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 1.4, 2.8],
    "soil_saturation_pct": [20, 25, 35, 45, 60, 70, 80, 95, 30, 65],
    "flood_risk": [
        "Low", "Low", "Low", "Moderate", "Moderate",
        "High", "High", "High", "Low", "Moderate"
    ]
}

df = pd.DataFrame(data)

X = df[[
    "rainfall_mm",
    "river_level_m",
    "soil_saturation_pct"
]]
y = df["flood_risk"]

# Train a demonstration model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
model.fit(X, y)

st.subheader("Environmental Conditions")

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

if st.button("Predict Risk"):
    new_data = pd.DataFrame([{
        "rainfall_mm": rainfall,
        "river_level_m": river_level,
        "soil_saturation_pct": soil_saturation
    }])

    prediction = model.predict(new_data)[0]

    st.subheader("Prediction Result")
    st.metric("Predicted category", prediction)
    st.dataframe(new_data, use_container_width=True)

st.divider()
st.subheader("About the Project")
st.write(
    "This project demonstrates how a Decision Tree "
    "Classifier can process environmental inputs. "
    "The sample dataset is synthetic and is not suitable "
    "for real-world emergency decisions."
)

with st.expander("View sample dataset"):
    st.dataframe(df, use_container_width=True)
