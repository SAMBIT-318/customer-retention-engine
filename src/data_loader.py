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

    def get_train_test_split(self, df: pd.DataFrame, target_col: str, test_size=0.2):
        """Splits data into training and testing sets."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
