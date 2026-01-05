import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent))
import config

class DatasetManager:
    def __init__(self):
        self.data: pd.DataFrame = None
        self.problem_type: str = "classification" # Default
        
    def load_data(self, filename: str) -> pd.DataFrame:
        file_path = config.DATA_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"❌ File {filename} not found")
        
        self.data = pd.read_csv(file_path)
        return self.data

    def preprocess(self, target_column: str, drop_cols: list = None) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        Returns: X, y, problem_type
        """
        df = self.data.copy()

        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')
            
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # --- AUTO-DETECT PROBLEM TYPE ---
        # Logic: If target is numeric and has many unique values -> Regression
        # If target is string OR has few unique values (like 0,1 or 1-5) -> Classification
        
        is_numeric = pd.api.types.is_numeric_dtype(y)
        unique_count = y.nunique()
        
        if not is_numeric:
            self.problem_type = "classification"
        elif unique_count < 20: 
            # Heuristic: Less than 20 unique numbers usually means categories/ranking
            self.problem_type = "classification"
        else:
            self.problem_type = "regression"

        print(f"🕵️ Detected Problem Type: {self.problem_type}")

        # Auto-detect features
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

        # Handle Missing Values
        for col in self.numerical_cols:
            X[col] = X[col].fillna(X[col].median())
        
        for col in self.categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0])

        return X, y, self.problem_type