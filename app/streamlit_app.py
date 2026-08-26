import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Retention Engine", page_icon="🎯", layout="centered")

st.title("🎯 Customer Retention Engine")
st.write("Predict customer churn risk in real-time.")

# Load Model Safely
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_churn_model.pkl')
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading the model! Details: {e}")
        return None

model = load_model()

# User Inputs
st.markdown("### Customer Details")
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])

# Predict Button
if st.button("Predict Risk", type="primary"):
    if model is None:
        st.error("Model not loaded.")
    else:
        # Convert text inputs to the numbers the model expects
        contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        support_mapping = {"No": 0, "Yes": 1}
        
        input_data = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "Contract": [contract_mapping[contract]],
            "TechSupport": [support_mapping[tech_support]]
        })
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100
        
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **High Churn Risk!** ({probability:.1f}% chance of leaving)")
        else:
            st.success(f"✅ **Low Churn Risk** ({probability:.1f}% chance of leaving)")
