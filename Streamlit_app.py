import streamlit as st
import pickle
import numpy as np

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("📊 Student Final Score Prediction")
st.write("Linear Regression Model")
st.divider()

# ----------------------------------
# Load trained objects
# ----------------------------------
@st.cache_resource
def load_models():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_selector.pkl", "rb") as f:
        selector = pickle.load(f)
    with open("Students_final_score.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, selector, model

scaler, selector, model = load_models()

TOTAL_FEATURES = scaler.n_features_in_

# ----------------------------------
# ✅ Select only a few real features
# (Use column names from your dataset)
# ----------------------------------
SELECTED_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Previous_Score",
    "Assignments_Completed",
    "Mock_Test_Score"
]

NUM_FEATURES = len(SELECTED_FEATURES)

st.info(f"Using {NUM_FEATURES} selected features for prediction")

# ----------------------------------
# User Inputs (Integer only)
# ----------------------------------
st.subheader("Enter Feature Values")

inputs = []
for name in SELECTED_FEATURES:
    val = st.number_input(
        label=name,
        min_value=0,
        step=1,          # ✅ only whole numbers
        value=0
    )
    inputs.append(val)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔮 Predict Final Score"):
    try:
        # Pad remaining features with zeros
        if NUM_FEATURES < TOTAL_FEATURES:
            inputs_extended = inputs + [0] * (TOTAL_FEATURES - NUM_FEATURES)
        else:
            inputs_extended = inputs

        input_data = np.array([inputs_extended])

        # Preprocessing
        scaled_data = scaler.transform(input_data)
        selected_data = selector.transform(scaled_data)

        # Prediction
        prediction = model.predict(selected_data)

        st.success(f"🎯 Predicted Final Score: **{prediction[0]:.2f}**")

    except Exception as e:
        st.error("❌ Prediction failed. Check feature order and dimensions.")
        st.exception(e)
