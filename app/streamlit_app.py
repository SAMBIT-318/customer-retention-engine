import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# We changed layout to "wide" so the charts have more room to breathe!
st.set_page_config(page_title="Customer Retention Engine", page_icon="🎯", layout="wide")

st.title("🎯 Customer Retention Engine")

@st.cache_resource
def load_model():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    MODEL_PATH = os.path.join(ROOT_DIR, 'model', 'best_churn_model.pkl')
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return None

model = load_model()

# --- Create Tabs ---
tab1, tab2 = st.tabs(["👤 Single Predictor", "📊 Analytics Dashboard"])

# ==========================================
# TAB 1: YOUR ORIGINAL PREDICTOR
# ==========================================
with tab1:
    st.markdown("### Predict a Single Customer")
    
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0)
    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])

    if st.button("Predict Risk", type="primary"):
        if model is None:
            st.error("Cannot predict: Model is not loaded.")
        else:
            contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            support_mapping = {"No": 0, "Yes": 1}
            
            input_data = pd.DataFrame({
                "tenure": [tenure],
                "MonthlyCharges": [monthly_charges],
                "Contract": [contract_mapping[contract]],
                "TechSupport": [support_mapping[tech_support]]
            })
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100
            
            st.divider()
            if prediction == 1:
                st.error(f"⚠️ **High Churn Risk!**")
                st.write(f"The model estimates a **{probability:.1f}%** chance this customer will cancel.")
            else:
                st.success(f"✅ **Low Churn Risk**")
                st.write(f"The model estimates only a **{probability:.1f}%** chance this customer will cancel.")

# ==========================================
# TAB 2: ANALYTICS DASHBOARD
# ==========================================
with tab2:
    st.markdown("### 📊 Churn Trends & Analytics")
    
    # 1. Load the real CSV data safely using caching so it runs fast
    @st.cache_data
    def load_dashboard_data():
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.dirname(CURRENT_DIR)
        DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            # Fallback dummy data just in case the file isn't uploaded to GitHub yet
            st.warning("⚠️ Could not find real data file. Showing sample data.")
            return pd.DataFrame({
                "tenure": [1, 24, 72, 12, 60, 4, 36, 70],
                "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Two year", "Month-to-month", "One year", "Two year"],
                "Churn": ["Yes", "No", "No", "Yes", "No", "Yes", "No", "No"]
            })

    df = load_dashboard_data()

    # 2. Create Layout for Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Does Contract Type Affect Churn?**")
        # Create a Plotly Histogram showing Churn grouped by Contract Type
        fig_contract = px.histogram(
            df, 
            x="Contract", 
            color="Churn", 
            barmode="group",
            color_discrete_map={"Yes": "#ff4b4b", "No": "#00cc96"} # Streamlit red/green colors
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    with chart_col2:
        st.markdown("**When Do Customers Usually Leave?**")
        # Create a Plotly Box Plot showing the distribution of Tenure vs Churn
        fig_tenure = px.box(
            df, 
            x="Churn", 
            y="tenure", 
            color="Churn",
            color_discrete_map={"Yes": "#ff4b4b", "No": "#00cc96"}
        )
        st.plotly_chart(fig_tenure, use_container_width=True)
