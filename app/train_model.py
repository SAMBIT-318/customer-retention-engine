import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Create a "model" folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# 2. Create some sample training data
# These features MUST match what you ask for in your Streamlit app
data = {
    'tenure': [1, 24, 72, 12, 60, 4, 36, 70],
    'MonthlyCharges': [29.85, 56.95, 105.65, 42.30, 89.10, 74.40, 55.20, 110.50],
    # Contract: 0 = Month-to-month, 1 = One year, 2 = Two year
    'Contract': [0, 1, 2, 0, 2, 0, 1, 2],
    # TechSupport: 0 = No, 1 = Yes
    'TechSupport': [0, 1, 1, 0, 1, 0, 1, 1],
    # Target variable: 0 = Stayed, 1 = Churned
    'Churn': [1, 0, 0, 1, 0, 1, 0, 0] 
}

df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 3. Train the machine learning model
print("Training model...")
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 4. Save the model to a .pkl file
model_path = os.path.join("model", "best_churn_model.pkl")
joblib.dump(model, model_path)

print(f"✅ Success! Real model saved to: {model_path}")
