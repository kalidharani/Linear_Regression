import streamlit as st
import joblib
import numpy as np

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Student Final Score Prediction",
    layout="centered"
)

st.title("📊 Student Final Score Prediction")
st.write("Linear Regression Model")
st.divider()

# ----------------------------------
# Load trained objects (JOBLIB)
# ----------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    model = joblib.load("model.pkl")
    return scaler, selector, model


scaler, selector, model = load_models()

# ----------------------------------
# Feature configuration
# ----------------------------------
FEATURES = [
    "Previous_Sem_Score",
    "Study_Hours_per_Week",
    "Attendance_Percentage",
    "Family_Income",
    "Teacher_Feedback_en",
    "Sleep_Hours",
    "Internet_Access_en",
    "Motivation_Level",
    "Peer_Influence"
]

TOTAL_FEATURES = scaler.n_features_in_
NUM_FEATURES = len(FEATURES)

st.info(f"Using {NUM_FEATURES} features for prediction")

# ----------------------------------
# User Inputs (WHOLE NUMBERS ONLY)
# ----------------------------------
st.subheader("Enter Student Details")

inputs = []
for feature in FEATURES:
    value = st.number_input(
        label=feature,
        min_value=0,
        value=0,
        step=1,          # ✅ whole numbers only
        format="%d"
    )
    inputs.append(int(value))  # ✅ force integer

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔮 Predict Final Score"):
    try:
        # Ensure correct feature length
        if NUM_FEATURES < TOTAL_FEATURES:
            inputs_extended = inputs + [0] * (TOTAL_FEATURES - NUM_FEATURES)
        else:
            inputs_extended = inputs[:TOTAL_FEATURES]

        input_array = np.array([inputs_extended])

        # Preprocessing
        scaled_data = scaler.transform(input_array)
        selected_data = selector.transform(scaled_data)

        # Prediction
        prediction = model.predict(selected_data)

        st.success(f"🎯 Predicted Final Score: **{prediction[0]:.2f}**")

    except Exception as e:
        st.error("❌ Prediction failed due to feature mismatch or model issue.")
        st.exception(e)
