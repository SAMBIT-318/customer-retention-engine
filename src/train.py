# Assuming this class is saved in src/data_loader.py
from src.data_loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Initialize and load
loader = DataLoader('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
raw_df = loader.load_data()

# 2. Preprocess (Crucial new step!)
clean_df = loader.preprocess_data(raw_df)

# 3. Split
X_train, X_test, y_train, y_test = loader.get_train_test_split(clean_df, target_col='Churn')

# 4. Train and Save
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/best_churn_model.pkl')
