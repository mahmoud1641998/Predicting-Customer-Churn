import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Storing Columns for Encoding and Scaling
cat_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']  # Columns that need encoding

con_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']  # Columns that need scaling

# Loading Stored Objects for Reuse
cols = pickle.load(open(r"fetures.pkl", "rb"))
scaler = pickle.load(open(r"scaller.pkl", "rb"))
encoder = pickle.load(open(r"encoders.pkl", "rb"))
model = pickle.load(open(r"model.pkl", "rb"))


def churning(features, model):
    """
    Process input data, apply encoding & scaling, and predict churn.
    Returns prediction (0 or 1) and churn probability.
    """
    data = pd.DataFrame([features], columns=cols)

    # Encoding categorical features
    for col in cat_cols:
        data[col] = encoder[col].transform(data[col])

    # Scaling continuous features
    data[con_cols] = scaler.transform(data[con_cols])

    # Ensure data is a NumPy array with correct type
    data_array = np.array(data, dtype=np.float32)

    # Predict churn and probability
    predicted = model.predict(data_array)[0]
    probability = float(model.predict_proba(data_array)[0][1])  # Convert to Python float

    return predicted, probability


# Streamlit UI
st.title("📊 Customer Churn Prediction")

# Sidebar Inputs
st.sidebar.header("Enter Customer Details")

senior_citizen = st.sidebar.radio("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
phone_service = st.sidebar.radio("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.radio("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0, step=1.0)

# Convert categorical inputs to match model expectations
input_data = np.array([senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service,
                        online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies,
                        contract, paperless_billing, payment_method, monthly_charges, total_charges])

# Prediction Button
if st.sidebar.button("Predict Churn"):
    prediction, probability = churning(input_data, model)

    st.subheader("🔍 Prediction Result")
    st.write(f"**Customer is likely to {'Churn' if prediction == 1 else 'Stay'}**")

    # Show churn probability
    st.progress(float(probability))  # Ensure probability is a Python float

    # Display probability as a percentage
    st.write(f"📊 **Churn Probability: {probability * 100:.2f}%**")