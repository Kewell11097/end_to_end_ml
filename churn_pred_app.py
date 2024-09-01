import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

model = joblib.load("best_model1.pkl")

st.title("Bank Customer Churn Prediction App")

credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600, step=1)
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5, step=1)
balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=10000.0, step=100.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=300000.0, value=50000.0, step=100.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])

if st.button("Predict"):
    input_data = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_credit_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
            "Geography_Spain": [1 if geography == "Spain" else 0],  
        }
    )

    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.success("The custormer is at risk of churn.")
    else:
        st.success("THe customer is not at risk of churn.")