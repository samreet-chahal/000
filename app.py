# gender -> 1 Male and 0 Female
# Churn -> 1 Yes and 0 No
# order of X -> ['tenure', 'MonthlyCharges', 'gender_Male']

import streamlit as st
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="centered",
)

# ── Load model & scaler ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    scaler = joblib.load("scaler.pkl")
    model  = joblib.load("model.pkl")
    return scaler, model

scaler, model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title(" Telco Customer Churn Predictor")
st.caption("Enter a customer's details below to predict whether they are likely to churn.")
st.divider()

# ── Input form ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0, max_value=100, value=10,
        help="How long the customer has been with the company"
    )

with col2:
    monthly_charge = st.number_input(
        "Monthly Charge ($)",
        min_value=30, max_value=150, value=70,
        help="The customer's current monthly bill"
    )

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button(" Predict Churn", use_container_width=True, type="primary"):
    gender_male = 1 if gender == "Male" else 0

    X        = np.array([tenure, monthly_charge, gender_male]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction   = model.predict(X_scaled)[0]
    probability  = model.predict_proba(X_scaled)[0]   # [P(No churn), P(Churn)]
    churn_prob   = round(probability[1] * 100, 1)
    no_churn_prob = round(probability[0] * 100, 1)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"  This customer is **likely to churn** ({churn_prob}% probability)")
    else:
        st.success(f"  This customer is **not likely to churn** ({no_churn_prob}% probability of staying)")

    # Probability bar
    st.markdown("**Churn probability breakdown**")
    col_a, col_b = st.columns(2)
    col_a.metric(" Churn",     f"{churn_prob}%")
    col_b.metric(" No Churn",  f"{no_churn_prob}%")
    st.progress(int(churn_prob))

    # Input summary
    with st.expander(" Input summary"):
        st.write({
            "Gender":         gender,
            "Tenure (months)": tenure,
            "Monthly Charge": f"${monthly_charge}",
        })
else:
    st.info(" Fill in the customer details above and click **Predict Churn**.")
