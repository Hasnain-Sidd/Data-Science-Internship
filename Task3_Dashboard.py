import streamlit as st 
import joblib
import numpy as np 
import pandas as pd

# Load model and scaler
scalar = joblib.load('scalar3.pkl')
model = joblib.load('Model3.pkl')

# Page settings
st.set_page_config(page_title="Customer Churn Prediction 🛃", layout='wide')
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🛃 Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("#### 📋 Enter Customer Details Below:")

# Input layout using columns
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("💳 Credit Score", min_value=0)
    geography = st.selectbox("🌍 Geography", ['France', 'Germany', 'Spain'])
    gender = st.selectbox("👤 Gender", ['Male', 'Female'])
    age = st.number_input("🎂 Age", min_value=18)
    tenure = st.number_input("📆 Tenure (Years)", min_value=0)

with col2:
    balance = st.number_input("💰 Account Balance", min_value=0.0, format="%.2f")
    num_of_products = st.number_input("📦 Number of Products", min_value=0)
    has_card = st.selectbox("💳 Has Credit Card?", [0, 1])
    is_active_memeber = st.selectbox("🟢 Is Active Member?", [0, 1])
    estimated_salary = st.number_input("💼 Estimated Salary", min_value=0.0, format="%.2f")

st.markdown("---")
st.markdown("### 🔮 Click Below to Predict Churn")
prediction_button = st.button("🚀 Predict")

# Prediction logic
if prediction_button:
    gender_selected = 1 if gender == "Female" else 0
    geography_selected = {'France': 1, 'Spain': 2, 'Germany': 3}[geography]

    X = [credit_score, geography_selected, age, gender_selected, tenure,
         balance, num_of_products, has_card, is_active_memeber, estimated_salary]

    X_array = scalar.transform([np.array(X)])
    prediction = model.predict(X_array)
    predicted = '❌ Will Churn' if prediction == 1 else '✅ Will Not Churn'

    st.success(f"📊 Prediction Result: **{predicted}**")
else:
    st.info("🔎 Please fill in the form and click Predict.")
