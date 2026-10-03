import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.config import load_config, TimeSeriesConfig, ModelTrainingConfig

class ChronologicalPreprocessor:
    """Chronological dataset splitter and data leakage-free feature scaler."""

    def __init__(self, time_series_cfg: Optional[TimeSeriesConfig] = None, training_cfg: Optional[ModelTrainingConfig] = None):
        cfg = load_config()
        self.ts_cfg = time_series_cfg or cfg.time_series
        self.train_cfg = training_cfg or cfg.model_training
        self.paths = cfg.paths
        
        self.feature_columns = self.ts_cfg.feature_columns
        self.target_column = self.ts_cfg.target_column
        
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def load_raw_csv(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        path = csv_path or self.paths.traffic_csv
        if not os.path.exists(path):
            raise FileNotFoundError(f"Traffic CSV missing at: {path}")
        df = pd.read_csv(path)
        return df

    def split_chronologically(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits time-series chronologically into Train (70%), Validation (15%), Test (15%) without shuffling."""
        n = len(df)
        train_end = int(n * self.train_cfg.train_split)
        val_end = int(n * (self.train_cfg.train_split + self.train_cfg.val_split))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        print(f"[INFO] Chronological Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (Total={n})")
        return train_df, val_df, test_df

    def fit_transform_train(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fits scalers strictly on TRAIN data and transforms TRAIN features and target."""
        X_train_raw = train_df[self.feature_columns].values
        y_train_raw = train_df[[self.target_column]].values

        X_train_scaled = self.feature_scaler.fit_transform(X_train_raw)
        y_train_scaled = self.target_scaler.fit_transform(y_train_raw).flatten()

        self.is_fitted = True
        return X_train_scaled, y_train_scaled

    def transform_eval(self, eval_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms Validation/Test features and target using scalers fitted ONLY on TRAIN data."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming validation/test sets.")

        X_raw = eval_df[self.feature_columns].values
        y_raw = eval_df[[self.target_column]].values

        X_scaled = self.feature_scaler.transform(X_raw)
        y_scaled = self.target_scaler.transform(y_raw).flatten()

        return X_scaled, y_scaled

    def save_scalers(self, output_dir: Optional[str] = None):
        """Saves fitted scalers to disk."""
        out_dir = output_dir or self.paths.processed_data_dir
        os.makedirs(out_dir, exist_ok=True)
        
        feature_scaler_path = os.path.join(out_dir, "feature_scaler.pkl")
        target_scaler_path = os.path.join(out_dir, "target_scaler.pkl")

        joblib.dump(self.feature_scaler, feature_scaler_path)
        joblib.dump(self.target_scaler, target_scaler_path)
        print(f"[INFO] Scalers saved to {out_dir}")

    def load_scalers(self, input_dir: Optional[str] = None):
        """Loads fitted scalers from disk."""
        in_dir = input_dir or self.paths.processed_data_dir
        feature_scaler_path = os.path.join(in_dir, "feature_scaler.pkl")
        target_scaler_path = os.path.join(in_dir, "target_scaler.pkl")

        if not os.path.exists(feature_scaler_path) or not os.path.exists(target_scaler_path):
            raise FileNotFoundError(f"Scaler pickle files missing in {in_dir}")

        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)
        self.is_fitted = True
        print(f"[INFO] Scalers loaded successfully from {in_dir}")

if __name__ == "__main__":
    prep = ChronologicalPreprocessor()
    df = prep.load_raw_csv()
    tr, val, te = prep.split_chronologically(df)
    X_tr, y_tr = prep.fit_transform_train(tr)
    X_val, y_val = prep.transform_eval(val)
    print("Preprocessed Train X shape:", X_tr.shape, "Train y shape:", y_tr.shape)
    print("Preprocessed Val X shape:", X_val.shape, "Val y shape:", y_val.shape)
