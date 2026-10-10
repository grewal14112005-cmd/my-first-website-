
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
    "Not suitable for real-world flood warnings."
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

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

st.subheader("🌧️ Environmental Conditions")

col1, col2 = st.columns(2)

with col1:
    rainfall = st.number_input(
        "Rainfall (mm)", 0.0, 500.0, 85.0
    )

with col2:
    river_level = st.number_input(
        "River level (m)", 0.0, 20.0, 2.7
    )

soil_saturation = st.slider(
    "Soil saturation (%)", 0, 100, 72
)

if st.button("🔍 Predict Risk", type="primary"):
    new_data = pd.DataFrame([{
        "rainfall_mm": rainfall,
        "river_level_m": river_level,
        "soil_saturation_pct": soil_saturation
    }])

    prediction = model.predict(new_data)[0]

    st.subheader("Prediction Result")

    if prediction == "Low":
        st.success("🟢 LOW RISK CATEGORY")
        guidance = (
            "Continue monitoring conditions. "
            "Follow local weather and official advisories."
        )
    elif prediction == "Moderate":
        st.warning("🟠 MODERATE RISK CATEGORY")
        guidance = (
            "Monitor official weather alerts and river updates. "
            "Prepare an emergency kit as appropriate."
        )
    else:
        st.error("🔴 HIGH RISK CATEGORY")
        guidance = (
            "Check official emergency alerts immediately. "
            "Follow local authorities' evacuation instructions "
            "if issued. Do not enter floodwater."
        )

    st.info(
        "Illustrative model output only. "
        "Use official warnings for real safety decisions."
    )

    st.markdown("### Environmental readings")

    a, b, c = st.columns(3)
    a.metric("Rainfall", f"{rainfall:.1f} mm")
    b.metric("River level", f"{river_level:.2f} m")
    c.metric("Soil saturation", f"{soil_saturation}%")

    st.markdown("### Suggested response")
    st.write(guidance)

st.divider()

st.subheader("About the Project")
st.write(
    "This project demonstrates a Decision Tree Classifier "
    "using three environmental inputs. The dataset contains "
    "only 10 synthetic examples, so predictions are not "
    "scientifically validated and must not guide real "
    "emergency decisions."
)

with st.expander("View sample dataset"):
    st.dataframe(df, use_container_width=True)
    
