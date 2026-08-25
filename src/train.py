import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    print("Loading data...")
    # 1. Load Data (Make sure the CSV name exactly matches your downloaded file)
    file_path = '../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df = pd.read_csv(file_path)
    
    print("Preprocessing data...")
    # 2. Convert Target ('Churn') from Yes/No to 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 3. Convert 'Contract' text to numbers (0, 1, 2) to match our Streamlit app
    df['Contract'] = df['Contract'].map({
        'Month-to-month': 0, 
        'One year': 1, 
        'Two year': 2
    })
    
    # 4. Select ONLY the features our Streamlit app uses
    features = ['tenure', 'MonthlyCharges', 'Contract']
    X = df[features]
    y = df['Churn']
    
    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Initialize and Train Model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 7. Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, preds))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, probs):.4f}")
    
    # 8. Save Model
    # Ensure the models directory exists
    os.makedirs('../models', exist_ok=True) 
    joblib.dump(model, '../models/best_churn_model.pkl')
    print("\n✅ Success! Model saved to models/best_churn_model.pkl")

if __name__ == "__main__":
    train_model()
