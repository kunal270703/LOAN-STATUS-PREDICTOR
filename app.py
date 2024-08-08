import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('loan_status_prediction (1).pkl')
scaler = joblib.load('scaler3 (2).pkl')


def predict_loan_status(inputs):
    inputs_scaled = scaler.transform([inputs])
    prediction = model.predict(inputs_scaled)
    return prediction[0]


def main():
    st.title("Loan Status Prediction")

    # Input fields for each feature
    gender = st.selectbox("Gender", options=["Male", "Female"])
    marital_status = st.selectbox("Marital Status", options=["Married", "Single"])
    dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
    application_income = st.number_input("Application Income", min_value=0.0, step=1000.0)
    co_application_income = st.number_input("Co-Application Income", min_value=0.0, step=1000.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=1000.0)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0, step=1)
    credit_history = st.selectbox("Credit History", options=["0", "1"])
    property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

    # Convert categorical input features to numerical format if needed
    gender = 1 if gender == "Male" else 0
    marital_status = 1 if marital_status == "Married" else 0
    dependents = 3 if dependents == "3+" else int(dependents)
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = int(credit_history)
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    if st.button("Predict"):
        features = [
            gender,
            marital_status,
            dependents,
            education,
            self_employed,
            application_income,
            co_application_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area
        ]
        result = predict_loan_status(features)
        st.write(f"{'LOAN IS APPROVED' if result == 1 else 'LOAN IS REJECTED'}")


if __name__ == "__main__":
    main()
