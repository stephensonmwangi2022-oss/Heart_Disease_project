import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler, OrdinalEncoder
class DataPreprocessor:
    def __init__(self, fill_strategies=None, encoding_strategies=None):
        self.fill_strategies = fill_strategies or {}
        self.train_fill_values = {}
        self.encoding_strategies = encoding_strategies or {}
        self.encoders = {}
        self.numerical_cols = None  # will be set in fit
        self.scaler = None

    def fit(self, X):
        X = X.copy()
        # Determine numerical columns from X itself
        self.numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        
        # Fill missing values
        for col, strategy in self.fill_strategies.items():
            if strategy == "mean":
                self.train_fill_values[col] = X[col].mean()
            elif strategy == "median":
                self.train_fill_values[col] = X[col].median()
            elif strategy == "mode":
                self.train_fill_values[col] = X[col].mode()[0]
            elif strategy == "unknown":
                self.train_fill_values[col] = "unknown"
            else:
                self.train_fill_values[col] = strategy

        # Fit encoders
        for col, strategy in self.encoding_strategies.items():
            if col not in X.columns:
                continue  # skip if column not in dataset (like 'dataset' column)
            if strategy == "onehot":
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                enc.fit(X[[col]])
            elif strategy == "ordinal":
                enc = OrdinalEncoder()
                enc.fit(X[[col]])
            self.encoders[col] = enc

        # Fit scaler on numerical columns
        self.scaler = StandardScaler()
        self.scaler.fit(X[self.numerical_cols])

    def transform(self, X):
        X = X.copy()
        
        # Fill missing
        for col, value in self.train_fill_values.items():
            if col not in X.columns:
                continue
            if value == "unknown":
                X[col] = X[col].fillna("unknown")
            else:
                X[col] = X[col].fillna(value)
        
        # Encode categorical
        for col in self.encoding_strategies.keys():
            if col not in X.columns:
                continue
            X[col] = X[col].astype(str)
            enc = self.encoders.get(col)
            if enc is None:
                continue
            if isinstance(enc, OneHotEncoder):
                encoded = pd.DataFrame(
                    enc.transform(X[[col]]),
                    columns=enc.get_feature_names_out([col]),
                    index=X.index
                )
                X = X.drop(columns=[col])
                X = pd.concat([X, encoded], axis=1)
            else:
                X[col] = enc.transform(X[[col]])

        # Scale numerical columns
        X[self.numerical_cols] = self.scaler.transform(X[self.numerical_cols])
        return X
