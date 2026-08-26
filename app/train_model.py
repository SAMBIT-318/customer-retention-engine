import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Safely locate the root directory (one level up from the 'app' folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# 2. Define the correct model directory and ensure it exists
MODEL_DIR = os.path.join(ROOT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# 3. Create sample training data
data = {
    'tenure': [1, 24, 72, 12, 60, 4, 36, 70],
    'MonthlyCharges': [29.85, 56.95, 105.65, 42.30, 89.10, 74.40, 55.20, 110.50],
    'Contract': [0, 1, 2, 0, 2, 0, 1, 2],
    'TechSupport': [0, 1, 1, 0, 1, 0, 1, 1],
    'Churn': [1, 0, 0, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 4. Train the model
print("Training the model...")
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 5. Save the trained model to the root 'model/' folder
model_path = os.path.join(MODEL_DIR, "best_churn_model.pkl")
joblib.dump(model, model_path)

print(f"✅ Success! The real binary model has been saved to: {model_path}")
