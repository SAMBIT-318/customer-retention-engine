import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------------------------------------------------------
# 1. Page Configuration & UI Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Customer Retention Engine", page_icon="🎯", layout="centered")

st.title("🎯 Customer Retention Engine")
st.write("Predict customer churn risk in real-time.")

# -----------------------------------------------------------------------------
# 2. Robust Model Loading
# -----------------------------------------------------------------------------
# @st.cache_resource prevents Streamlit from reloading the model every time the user clicks a button
@st.cache_resource
def load_model():
    try:
        # Construct the absolute path to the model file safely
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_churn_model.pkl')
        
        # Load and return the model
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model! Details: {e}")
        return None

# Load the model into memory
model = load_model()

# -----------------------------------------------------------------------------
# 3. User Input Form
# -----------------------------------------------------------------------------
st.markdown("### Customer Details")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    # REPLACE THESE with the actual features your model expects!
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0)

with col2:
    # Example categorical features
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])

# -----------------------------------------------------------------------------
# 4. Data Preprocessing & Prediction
# -----------------------------------------------------------------------------
if st.button("Predict Risk", type="primary"):
    
    if model is None:
        st.error("Cannot make a prediction because the model failed to load.")
    else:
        # Step A: Convert categorical inputs into numbers if your model requires it
        # (For example, turning 'Yes'/'No' into 1/0)
        contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        support_mapping = {"No": 0, "Yes": 1}
        
        # Step B: Organize the inputs into a pandas DataFrame exactly how the model expects it
        # The column names MUST match the column names used during model training
        input_data = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "Contract": [contract_mapping[contract]],
            "TechSupport": [support_mapping[tech_support]]
        })
        
        # Step C: Run the prediction
        try:
            # Get the prediction (usually 0 for 'Stay' and 1 for 'Churn')
            prediction = model.predict(input_data)[0]
            
            # (Optional) Get the probability percentage if your model supports it
            probability = model.predict_proba(input_data)[0][1] * 100
            
            # Step D: Display the results
            st.divider()
            if prediction == 1:
                st.error(f"⚠️ **High Churn Risk!**")
                st.write(f"The model estimates a **{probability:.1f}%** chance this customer will cancel.")
            else:
                st.success(f"✅ **Low Churn Risk**")
                st.write(f"The model estimates only a **{probability:.1f}%** chance this customer will cancel.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction. Check that your input features match your model's training data. Details: {e}")
