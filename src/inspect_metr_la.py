import os
import pandas as pd
import numpy as np

def inspect_metr_la(csv_path: str = "data/raw/forecasting/METR-LA.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"METR-LA CSV missing at: {csv_path}")

    print(f"[INFO] Reading METR-LA CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    n_rows, n_cols = df.shape
    cols = list(df.columns)
    
    print("\n=== METR-LA DATASET INSPECTION ===")
    print(f"Shape: {n_rows} rows x {n_cols} columns")
    print(f"First 5 column names: {cols[:5]}")
    print(f"Last 5 column names: {cols[-5:]}")
    
    # Check if first column is timestamp/datetime
    first_col = cols[0]
    print(f"Primary index column: '{first_col}'")
    
    # Format determination: Wide vs Long
    is_wide = n_cols > 10
    print(f"Data format: {'WIDE format (sensors as columns)' if is_wide else 'LONG format'}")
    
    # Sensor columns (excluding timestamp column if present)
    sensor_cols = [c for c in cols if c != first_col]
    print(f"Number of unique sensors: {len(sensor_cols)}")

    # Parse timestamps
    df[first_col] = pd.to_datetime(df[first_col])
    ts_start = df[first_col].min()
    ts_end = df[first_col].max()
    
    # Compute sampling interval
    time_diffs = df[first_col].diff().dropna()
    sampling_interval = time_diffs.mode()[0] if not time_diffs.empty else None

    # Missing values
    total_missing = df.isnull().sum().sum()
    
    # Statistical summary across all sensor readings
    sensor_data = df[sensor_cols].values
    min_val = float(np.nanmin(sensor_data))
    max_val = float(np.nanmax(sensor_data))
    mean_val = float(np.nanmean(sensor_data))
    std_val = float(np.nanstd(sensor_data))

    print(f"Timestamp Range: {ts_start} to {ts_end}")
    print(f"Sampling Interval: {sampling_interval}")
    print(f"Total Missing Values: {total_missing}")
    print(f"Speed Statistics across all sensors -> Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}, Std: {std_val:.2f}")

    print("\nFirst 5 rows (first 6 columns):")
    print(df.iloc[:5, :6])

if __name__ == "__main__":
    inspect_metr_la()
