# gender -> 1 Male and 0 Female
# Churn -> 1 Yes and 0 No
# order of X -> ['tenure', 'MonthlyCharges', 'gender_Male']

import streamlit as st
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button to get a prediction.")

st.divider()

# Inputs
gender = st.selectbox("Enter Gender", ["Male", "Female"])
tenure = st.number_input("Enter tenure (months)", min_value=0, max_value=100, value=10)
monthlycharge = st.number_input("Enter monthly charge", min_value=30, max_value=150, value=70)

st.divider()

predictionbutton = st.button("Predict")

if predictionbutton:
    # Encode gender (Male = 1, Female = 0)
    gender_male = 1 if gender == "Male" else 0

    # Order MUST match training
    X = [tenure, monthlycharge, gender_male]

    X_array = np.array(X).reshape(1, -1)

    X_scaled = scaler.transform(X_array)

    prediction = model.predict(X_scaled)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.success(f"Predicted Churn: {predicted}")

else:
    st.info("Please enter the values and click Predict")