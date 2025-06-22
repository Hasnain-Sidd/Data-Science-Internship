import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import streamlit as st

# App Configuration
st.set_page_config(
    page_title="📊 Insurance Claim Amount Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center; color: teal;'>💼 Insurance Claim Amount Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Predict medical insurance claim amount using ML models")

# Load model and scaler
model = joblib.load("bagging_regressor.pkl")
scaler = joblib.load("scalar1.pkl")

# Sidebar Input Section
with st.sidebar:
    st.header("🔧 Input Parameters")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Gender", ['female', 'male'])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ['yes', 'no'])
    region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])
    prediction_button = st.button("🚀 Predict Claim Amount")

# Tabs for navigation
tab1, tab2 = st.tabs(["🧮 Prediction", "📈 Model Performance"])

# Prediction Tab
with tab1:
    if prediction_button:
        gender_selected = 1 if sex == 'female' else 0
        smoker_selected = 1 if smoker == 'yes' else 0
        region_selected = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}[region]

        X = [age, gender_selected, bmi, children, smoker_selected, region_selected]
        X_array = scaler.transform([X])
        prediction = model.predict(X_array)

        st.success(f"💰 Predicted Claim Amount: **${prediction[0]:,.2f}**")
        st.markdown("---")
    else:
        st.info("ℹ️ Enter the details in the sidebar and click **Predict Claim Amount**.")

# Model Performance Tab
with tab2:
    st.subheader("📊 Model Comparison Visualizations")
    st.markdown("Below are the performance comparison graphs of six machine learning models used:")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🔹 Linear Regression")
        st.image("linear_regression_graph.png")
    with col2:
        st.markdown("#### 🔹 Decision Tree")
        st.image("decision_tree_graph.png")
    with col3:
        st.markdown("#### 🔹 Random Forest")
        st.image("random_forest_graph.png")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("#### 🔹 Bagging Regressor")
        st.image("bagging_regressor_graph.png")
    with col5:
        st.markdown("#### 🔹 R-Squared Comparison")
        st.image("r_squared.png")
    with col6:
        st.markdown("#### 🔹 Mean Squared Comparison")
        st.image("mean_squared.png")

    st.markdown("---")
    st.info("✅ The model with the lowest MSE and highest R² (e.g., Bagging Regressor) is selected for prediction.")

# Footer
st.markdown(
    "<hr style='border:1px solid #ccc;'>"
    "<div style='text-align: center;'>"
    "Made with ❤️ by Hasnain Hissam | 📊 Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)
