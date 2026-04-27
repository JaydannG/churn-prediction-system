import streamlit as st
import pandas as pd
import joblib

from src.config import MODEL_PATH, BEST_THRESHOLD


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_risk_level(probability):
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def get_recommendation(probability):
    if probability >= 0.70:
        return "Offer retention discount, prioritize support outreach, and encourage long-term contract."
    elif probability >= 0.40:
        return "Monitor customer and consider targeted engagement or service bundle."
    return "Customer appears low risk. Continue normal engagement."


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Prediction App")
st.write(
    "Enter customer information below to estimate the probability that the customer will churn."
)

model = load_model()

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=75.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=float(monthly_charges * tenure)
)

customer_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([customer_data])

st.subheader("Customer Profile")
st.dataframe(input_df)

if st.button("Predict Churn"):
    churn_probability = model.predict_proba(input_df)[:, 1][0]
    churn_prediction = int(churn_probability >= BEST_THRESHOLD)

    risk_level = get_risk_level(churn_probability)
    recommendation = get_recommendation(churn_probability)

    st.subheader("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{churn_probability:.2%}"
    )

    st.write(f"**Prediction:** {'Likely to Churn' if churn_prediction == 1 else 'Not Likely to Churn'}")
    st.write(f"**Risk Level:** {risk_level}")
    st.write(f"**Recommended Action:** {recommendation}")

    st.progress(float(churn_probability))