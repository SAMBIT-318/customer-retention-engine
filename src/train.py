# 1. FIX: Changed import since train.py and data_loader.py are in the same folder
from data_loader import DataLoader 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main():
    print("Starting training process...")
    
    # 1. Initialize and load
    # This path assumes you are running the script from the root project folder
    loader = DataLoader('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    raw_df = loader.load_data()
    
    if raw_df is None:
        print("Stopping execution because data failed to load.")
        return

    # 2. Preprocess 
    clean_df = loader.preprocess_data(raw_df)

    # 3. Split
    X_train, X_test, y_train, y_test = loader.get_train_test_split(clean_df, target_col='Churn')

    # 4. Train 
    print("Training RandomForest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate (NEW: So you know if your model is actually good!)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2%}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 6. Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_churn_model.pkl')
    print("\nModel saved successfully to 'models/best_churn_model.pkl'")

# This ensures the code only runs if you execute this file directly
if __name__ == "__main__":
    main()
