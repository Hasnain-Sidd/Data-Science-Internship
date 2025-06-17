import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st
import joblib

# Load scaler and model
scalar = joblib.load('scalar.pkl')
model = joblib.load('Model1.pkl')

# Page Config
st.set_page_config("Credit Risk Prediction 💳", layout='centered')
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💳 Credit Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input UI
st.subheader("📋 Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", min_value=18, max_value=100)
    income = st.number_input("💰 Income ($)", min_value=0)
    home = st.selectbox("🏠 Home Ownership", ['RENT', 'OWN', 'MORTGAGE', "OTHER"])
    employer_length = st.number_input("⏳ Employment Length (Years)", min_value=0)
    person_length = st.number_input("📆 Credit History Length (Years)", min_value=0)

with col2:
    loan_intent = st.selectbox("🎯 Loan Purpose", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_grad = st.selectbox("🏷️ Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    loan_amount = st.number_input("💸 Loan Amount ($)", min_value=0)
    loan_int_rate = st.number_input("📈 Interest Rate (%)", min_value=0.0, format="%.2f")
    loan_percent_income = st.number_input("📊 Loan % of Income", min_value=0.0, format="%.2f")
    person_history = st.selectbox("📉 Previous Default History", ['Y', 'N'])

st.markdown("---")
st.markdown("### 🔍 Click Below to Predict Credit Risk")
prediction_button = st.button("🚀 Predict")

# Encode and Predict
if prediction_button:
    home_selected = {'RENT': 1, 'OWN': 2, 'MORTGAGE': 3, 'OTHER': 4}[home]
    loan_intent_selected = {
        "PERSONAL": 1, "EDUCATION": 2, "MEDICAL": 3,
        "VENTURE": 4, "HOMEIMPROVEMENT": 5, "DEBTCONSOLIDATION": 6
    }[loan_intent]
    loan_grad_selected = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}[loan_grad]
    person_history_selected = 1 if person_history == 'Y' else 2

    X = [age, income, home_selected, employer_length, loan_intent_selected, loan_grad_selected,
         loan_amount, loan_int_rate, loan_percent_income, person_history_selected, person_length]
    
    X_array = scalar.transform([X])
    prediction = model.predict(X_array)
    
    predicted = "❌ **Default**" if prediction == 1 else "✅ **No Default**"
    
    st.markdown("---")
    st.success(f"📊 Prediction Result: {predicted}")
else:
    st.info("🔎 Please fill out the form and click Predict.")
