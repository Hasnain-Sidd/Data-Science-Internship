# 🛃 Customer Churn Prediction App - Task 3

Welcome to my end-to-end **Customer Churn Prediction** project, where I built a complete machine learning pipeline to predict whether a customer will churn (leave) or stay with the bank — based on demographic and transactional features.
---
## 📌 Project Overview
This project includes:
✅ Data cleaning and feature engineering  
✅ Exploratory Data Analysis (EDA)  
✅ Machine learning model building and tuning  
✅ A fully interactive Streamlit web app for real-time churn prediction  
✅ Model and scaler serialization with `joblib`  

---
## 📂 Dataset
- **Source**: Kaggle – [Churn Modelling Dataset](https://www.kaggle.com/datasets)
- **Rows**: 10,000 customers  
- **Target**: `Exited` (1 = churned, 0 = retained)
---
## 🧪 Features Used
| Feature            | Description                          |
|--------------------|--------------------------------------|
| CreditScore        | Credit score of the customer         |
| Geography          | Country (France, Spain, Germany)     |
| Gender             | Male or Female                       |
| Age                | Age of customer                      |
| Tenure             | Years with the bank                  |
| Balance            | Account balance                      |
| NumOfProducts      | Number of bank products used         |
| HasCrCard          | Has credit card (1 = Yes, 0 = No)    |
| IsActiveMember     | Active member (1 = Yes, 0 = No)      |
| EstimatedSalary    | Customer’s estimated salary          |

---
## 📊 Exploratory Data Analysis
Some key insights from visualizations:
- **Females** tend to churn more than males.
- **Older customers** are more likely to leave.
- Customers with **fewer years of tenure** are more prone to churn.

EDA was done using `matplotlib` and `seaborn`.
---
## 🤖 Machine Learning Models
| Model                | Notes                                     |
|----------------------|-------------------------------------------|
| Logistic Regression  | Simple baseline model                     |
| KNN Classifier       | Tuned using GridSearchCV                  |
| Random Forest        | Best model with highest accuracy          |

### ✅ Final Model: Random Forest Classifier
- Tuned with `GridSearchCV`
- Best Accuracy: ~**86%** on test data
- Saved as `Model.pkl`
---
## 🧪 Preprocessing Steps
- Categorical encoding for `Gender` and `Geography`
- Feature scaling using `StandardScaler`
- Train-Test split (70% / 30%)
---
## 🌐 Web Application - Streamlit
The app allows real-time prediction of customer churn based on user inputs.
### How to Run the App:
```bash
streamlit run app.py
