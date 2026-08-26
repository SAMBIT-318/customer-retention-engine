import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

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

# --- Create 3 Tabs ---
tab1, tab2, tab3 = st.tabs(["👤 Single Predictor", "📊 Analytics Dashboard", "📁 Batch Predictor"])

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
    
    @st.cache_data
    def load_dashboard_data():
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.dirname(CURRENT_DIR)
        DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            return pd.DataFrame({
                "tenure": [1, 24, 72, 12, 60],
                "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Two year"],
                "Churn": ["Yes", "No", "No", "Yes", "No"]
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
# TAB 3: BATCH PREDICTOR (NEW!)
# ==========================================
with tab3:
    st.markdown("### 📁 Batch Churn Prediction")
    st.info("Upload a CSV file to predict churn risk for thousands of customers at once.")
    
    # 1. File Uploader Widget
    uploaded_file = st.file_uploader("Upload your customer data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # 2. Read the uploaded file
            batch_df = pd.read_csv(uploaded_file)
            st.write("🔍 **Preview of uploaded data:**")
            st.dataframe(batch_df.head(3)) # Show the first 3 rows
            
            # Check if it has the required columns
            required_cols = ['tenure', 'MonthlyCharges', 'Contract', 'TechSupport']
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"❌ Your CSV is missing these required columns: {', '.join(missing_cols)}")
                st.write("Make sure your CSV headers exactly match: `tenure`, `MonthlyCharges`, `Contract`, `TechSupport`")
            else:
                if st.button("🚀 Analyze All Customers", type="primary"):
                    with st.spinner("Analyzing risk for all customers..."):
                        
                        # 3. Clean and map the data exactly like we did in Tab 1
                        process_df = batch_df[required_cols].copy()
                        
                        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
                        tech_support_map = {'No': 0, 'Yes': 1, 'No internet service': 0}
                        
                        # Apply mapping if the columns contain text
                        if process_df['Contract'].dtype == 'O':
                            process_df['Contract'] = process_df['Contract'].map(contract_map)
                        if process_df['TechSupport'].dtype == 'O':
                            process_df['TechSupport'] = process_df['TechSupport'].map(tech_support_map)
                        
                        # Ensure everything is a number and fill any empty cells with 0
                        process_df['tenure'] = pd.to_numeric(process_df['tenure'], errors='coerce').fillna(0)
                        process_df['MonthlyCharges'] = pd.to_numeric(process_df['MonthlyCharges'], errors='coerce').fillna(0)
                        
                        # 4. Make Batch Predictions
                        predictions = model.predict(process_df)
                        probabilities = model.predict_proba(process_df)[:, 1]
                        
                        # 5. Attach results to the user's original dataframe
                        results_df = batch_df.copy()
                        results_df['Predicted_Churn'] = ['Yes (High Risk)' if p == 1 else 'No (Low Risk)' for p in predictions]
                        results_df['Churn_Probability_%'] = (probabilities * 100).round(1)
                        
                        st.success("✅ Analysis Complete!")
                        st.write("📊 **Final Results:**")
                        st.dataframe(results_df)
                        
                        # 6. Create a downloadable CSV button
                        csv_export = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Full Report as CSV",
                            data=csv_export,
                            file_name="Customer_Churn_Predictions.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Could not process the file. Error: {e}")
