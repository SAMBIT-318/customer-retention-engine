import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from data_loader import DataLoader

def train_model():
    # 1. Load Data (Replace 'churn_data.csv' with your actual dataset)
    loader = DataLoader('../data/raw/churn_data.csv')
    df = loader.load_data()
    
    # 2. Split Data (Assuming target column is named 'Churn')
    # Note: Ensure you handle preprocessing (encoding/scaling) before this in a real project!
    X_train, X_test, y_train, y_test = loader.get_train_test_split(df, target_col='Churn')
    
    # 3. Initialize and Train Model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, preds))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, probs):.4f}")
    
    # 5. Save Model
    joblib.dump(model, '../models/best_churn_model.pkl')
    print("Model saved to models/best_churn_model.pkl")

if __name__ == "__main__":
    # Uncomment to run training when you have your data ready
    # train_model()
    pass