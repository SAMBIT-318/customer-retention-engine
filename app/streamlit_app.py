import streamlit as st
import joblib
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    # Updated path to exactly match your GitHub repository structure:
    # Folder is 'model' (singular) and file has the '.txt' extension
    return joblib.load('model/best_churn_model.pkl')

st.title("🎯 Customer Retention Engine")
st.write("Predict customer churn risk in real-time.")

# Sidebar for User Inputs
st.sidebar.header("Customer Details")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 10.0, 150.0, 50.0)
contract_type = st.sidebar.selectbox("Contract Type", [0, 1, 2]) # e.g., 0: M2M, 1: 1Yr, 2: 2Yr

if st.button("Predict Risk"):
    # Create input dataframe (Make sure this matches your model's exact features)
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'Contract': [contract_type]
    })
    
    try:
        model = load_model()
        prediction = model.predict(input_data)[0]
        
        # Check if the model supports predict_proba (like Logistic Regression, Random Forest, etc.)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
            prob_text = f" (Probability: {probability:.1%})"
        else:
            prob_text = ""
            
        if prediction == 1:
            st.error(f"High Risk of Churn!{prob_text}")
        else:
            st.success(f"Customer is likely to stay.{prob_text}")
            
    except Exception as e:
        st.warning("Error loading or running the model! Details: " + str(e))
