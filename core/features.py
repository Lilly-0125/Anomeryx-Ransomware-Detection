import numpy as np
import pandas as pd


class FeatureExtractor:

    def calculate_file_entropy(self, file_path: str) -> float:
        try:
            with open(file_path, "rb") as f:
                data = f.read(65536)
            if not data:
                return 0.0
            byte_counts   = np.bincount(
                np.frombuffer(data, dtype=np.uint8), minlength=256)
            probabilities = byte_counts / len(data)
            probabilities = probabilities[probabilities > 0]
            entropy       = -np.sum(probabilities * np.log2(probabilities))
            return round(float(entropy), 4)
        except Exception:
            return 0.0

    def preprocess_dataset(self, df: pd.DataFrame,
                           label_col: str) -> tuple:
        feature_cols = [
            c for c in df.columns
            if c != label_col and
            df[c].dtype in ["float64", "int64", "float32", "int32"]
        ]
        X = df[feature_cols].copy().fillna(0)
        y = df[label_col].copy()
        if y.dtype == object:
            y = (y.str.lower() != "benign").astype(int)
        return X.values, y.values, feature_cols
