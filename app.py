import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

# Load trained keras model
model = keras.models.load_model("loan_model.h5")

# Load the saved StandardScaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🏦 Loan Approval Prediction App")

st.write("Enter applicant details to check if the loan will be approved or rejected.")

# Input fields
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=1)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0)

if st.button("Predict"):
    # Convert categorical values to numeric
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0

    # Feature vector
    features = np.array([[
        no_of_dependents,
        education_val,
        self_employed_val,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], dtype=float)

    # ✅ Scale features using the same StandardScaler from training
    features_scaled = scaler.transform(features)

    # Predict probability (sigmoid output)
    proba = model.predict(features_scaled)[0][0]
    approval_prob = round(proba * 100, 2)
    rejection_prob = round((1 - proba) * 100, 2)

    if proba >= 0.7:
        st.success(f"✅ Loan Approved! (Confidence: {approval_prob}%)")
        st.write(f"🔹 Rejection Probability: {rejection_prob}%")
    else:
        st.error(f"❌ Loan Rejected! (Confidence: {rejection_prob}%)")
        st.write(f"🔹 Approval Probability: {approval_prob}%")
