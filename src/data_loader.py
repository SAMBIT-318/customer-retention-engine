import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Loads the dataset from the specified path."""
        try:
            df = pd.read_csv(self.file_path)
            print(f"Data successfully loaded with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans data and encodes strings to integers for modeling."""
        # Create a copy to avoid modifying the original dataframe in-place
        df_clean = df.copy()

        # Map strings to numbers to match your Streamlit app inputs
        df_clean['Contract'] = df_clean['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
        df_clean['Churn'] = df_clean['Churn'].map({'No': 0, 'Yes': 1})

        # Drop rows with missing values in our target columns
        df_clean = df_clean.dropna(subset=['Contract', 'Churn', 'tenure', 'MonthlyCharges'])

        # Optional but recommended: Filter down to ONLY the features your Streamlit app uses
        # This prevents shape mismatch errors during prediction
        df_clean = df_clean[['tenure', 'MonthlyCharges', 'Contract', 'Churn']]
        
        return df_clean

    def get_train_test_split(self, df: pd.DataFrame, target_col: str, test_size=0.2):
        """Splits data into training and testing sets."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # Stratify=y ensures both train and test sets have the same ratio of churned vs non-churned customers
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
