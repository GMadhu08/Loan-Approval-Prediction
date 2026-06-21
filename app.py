import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏦 Loan Approval Prediction")

dependents = st.number_input("Number of Dependents", min_value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
income = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
cibil_score = st.number_input("CIBIL Score")
residential_assets = st.number_input("Residential Assets Value")
commercial_assets = st.number_input("Commercial Assets Value")
luxury_assets = st.number_input("Luxury Assets Value")
bank_assets = st.number_input("Bank Asset Value")

education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

if st.button("Predict"):

    data = pd.DataFrame({
        'no_of_dependents':[dependents],
        'education':[education],
        'self_employed':[self_employed],
        'income_annum':[income],
        'loan_amount':[loan_amount],
        'loan_term':[loan_term],
        'cibil_score':[cibil_score],
        'residential_assets_value':[residential_assets],
        'commercial_assets_value':[commercial_assets],
        'luxury_assets_value':[luxury_assets],
        'bank_asset_value':[bank_assets]
    })

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 0:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")