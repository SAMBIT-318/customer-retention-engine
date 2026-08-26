import streamlit as st
import joblib
import os

# Construct the path dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_churn_model.pkl')

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    st.success("Model loaded successfully!")
    # Proceed with your st.button('Predict Risk') logic here
