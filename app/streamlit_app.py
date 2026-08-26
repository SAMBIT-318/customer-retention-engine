import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import google.generativeai as genai

# Page configuration
st.set_page_config(page_title="Customer Retention Engine", page_icon="🎯", layout="wide")

st.title("🎯 Customer Retention Engine")

# Load the machine learning model securely
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

# --- Create 4 Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Single Predictor", 
    "📊 Analytics", 
    "📁 Batch Predictor", 
    "🤖 AI Strategist"
])

# ==========================================
# TAB 1: SINGLE PREDICTOR
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
                st.error(f"⚠️ **High Churn Risk! ({probability:.1f}%)**")
            else:
                st.success(f"✅ **Low Churn Risk ({probability:.1f}%)**")

# ==========================================
# TAB 2: ANALYTICS DASHBOARD
# ==========================================
with tab2:
    st.markdown("### 📊 Churn Trends & Analytics")
    
    @st.cache_data
    def load_dashboard_data():
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.dirname(CURRENT_DIR)
        DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            # Fallback dummy data if the real CSV isn't found
            return pd.DataFrame({
                "tenure": [1, 24, 72, 12, 60, 4, 36, 70],
                "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Two year", "Month-to-month", "One year", "Two year"],
                "Churn": ["Yes", "No", "No", "Yes", "No", "Yes", "No", "No"]
            })

    df = load_dashboard_data()
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Does Contract Type Affect Churn?**")
        fig_contract = px.histogram(df, x="Contract", color="Churn", barmode="group",
                                    color_discrete_map={"Yes": "#ff4b4b", "No": "#00cc96"})
        st.plotly_chart(fig_contract, use_container_width=True)

    with chart_col2:
        st.markdown("**When Do Customers Usually Leave?**")
        fig_tenure = px.box(df, x="Churn", y="tenure", color="Churn",
                            color_discrete_map={"Yes": "#ff4b4b", "No": "#00cc96"})
        st.plotly_chart(fig_tenure, use_container_width=True)

# ==========================================
# TAB 3: BATCH PREDICTOR
# ==========================================
with tab3:
    st.markdown("### 📁 Batch Churn Prediction")
    st.info("Upload a CSV file to predict churn risk for thousands of customers at once.")
    
    uploaded_file = st.file_uploader("Upload your customer data (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_cols = ['tenure', 'MonthlyCharges', 'Contract', 'TechSupport']
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            else:
                if st.button("🚀 Analyze All Customers", type="primary"):
                    if model is None:
                        st.error("Model not loaded!")
                    else:
                        with st.spinner("Analyzing risk..."):
                            process_df = batch_df[required_cols].copy()
                            if process_df['Contract'].dtype == 'O':
                                process_df['Contract'] = process_df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
                            if process_df['TechSupport'].dtype == 'O':
                                process_df['TechSupport'] = process_df['TechSupport'].map({'No': 0, 'Yes': 1, 'No internet service': 0})
                            
                            process_df = process_df.fillna(0)
                            
                            predictions = model.predict(process_df)
                            probabilities = model.predict_proba(process_df)[:, 1]
                            
                            results_df = batch_df.copy()
                            results_df['Predicted_Churn'] = ['Yes (High Risk)' if p == 1 else 'No (Low Risk)' for p in predictions]
                            results_df['Churn_Probability_%'] = (probabilities * 100).round(1)
                            
                            st.success("✅ Analysis Complete!")
                            st.dataframe(results_df.head(10))
                            
                            csv_export = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Full Report as CSV", csv_export, "Customer_Churn_Predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ==========================================
# TAB 4: AI RETENTION STRATEGIST 
# ==========================================
with tab4:
    st.markdown("### 🤖 AI Retention Strategist")
    st.info("Instantly generate personalized retention emails and call scripts for high-risk customers.")
    
    st.markdown("#### Target Customer Profile")
    ai_col1, ai_col2 = st.columns(2)
    with ai_col1:
        ai_tenure = st.number_input("Customer Tenure (Months)", min_value=0, value=2, key="ai_ten")
        ai_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=95.0, key="ai_charge")
    with ai_col2:
        ai_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="ai_cont")
        ai_support = st.selectbox("Tech Support", ["No", "Yes"], key="ai_sup")
        
    if st.button("✨ Generate Retention Plan", type="primary"):
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("⚠️ API Key not found! Please configure your Streamlit Secrets.")
        else:
            api_key = st.secrets["GEMINI_API_KEY"]
            try:
                genai.configure(api_key=api_key)
                ai_model = genai.GenerativeModel('gemini-2.5-flash')
                
                prompt = f"""
                You are a senior customer retention expert for a telecommunications company. 
                A customer is currently at high risk of canceling their service. Here is their profile:
                - Tenure with us: {ai_tenure} months
                - Monthly Charges: ${ai_charges}
                - Contract Type: {ai_contract}
                - Uses Tech Support: {ai_support}
                
                Please generate:
                1. A catchy, empathetic email subject line.
                2. A short, highly personalized email offering a targeted discount or solution to convince them to stay.
                3. A brief 3-bullet-point script for our customer service agent to use if they call this customer.
                """
                
                with st.spinner("AI is typing a custom strategy..."):
                    response = ai_model.generate_content(prompt)
                    
                st.success("✅ AI Strategy Generated!")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Failed to connect to AI. Error: {e}")
