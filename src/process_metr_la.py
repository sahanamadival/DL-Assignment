import os
import pandas as pd
import numpy as np
from typing import Dict, Any

def process_and_validate_metr_la(
    raw_csv: str = "data/raw/forecasting/METR-LA.csv",
    output_csv: str = "data/processed/metr_la_forecasting.csv",
    selected_sensor: str = "773869"
) -> Dict[str, Any]:
    """Processes raw METR-LA dataset, selects a single sensor time-series, validates quality, and saves clean dataset."""
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw METR-LA CSV missing at: {raw_csv}")

    print(f"[INFO] Processing METR-LA for sensor '{selected_sensor}'...")
    df_raw = pd.read_csv(raw_csv)
    
    time_col = df_raw.columns[0]
    if selected_sensor not in df_raw.columns:
        raise ValueError(f"Sensor ID '{selected_sensor}' not found in METR-LA columns.")

    # Extract time and target sensor speed
    df_clean = pd.DataFrame({
        "timestamp": pd.to_datetime(df_raw[time_col]),
        "sensor_id": selected_sensor,
        "traffic_speed": df_raw[selected_sensor].astype(float)
    })

    # Sort chronologically just in case
    df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

    # Save to data/processed/metr_la_forecasting.csv
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_clean.to_csv(output_csv, index=False)
    print(f"[INFO] Processed forecasting dataset saved to: {output_csv} ({len(df_clean)} rows)")

    # Compute validation statistics
    n_rows = len(df_clean)
    ts_start = df_clean["timestamp"].min()
    ts_end = df_clean["timestamp"].max()
    
    diffs = df_clean["timestamp"].diff().dropna()
    sampling_interval_str = str(diffs.mode()[0]) if not diffs.empty else "Unknown"

    missing_vals = int(df_clean["traffic_speed"].isnull().sum())
    duplicate_ts = int(df_clean["timestamp"].duplicated().sum())

    speeds = df_clean["traffic_speed"].values
    min_speed = float(np.min(speeds))
    max_speed = float(np.max(speeds))
    mean_speed = float(np.mean(speeds))
    std_speed = float(np.std(speeds))

    # Chronological Split Sizes (70% Train, 15% Val, 15% Test)
    n_train = int(n_rows * 0.70)
    n_val = int(n_rows * 0.15)
    n_test = n_rows - (n_train + n_val)

    return {
        "dataset_loaded": True,
        "raw_csv_shape": df_raw.shape,
        "raw_columns_count": len(df_raw.columns),
        "selected_sensor": selected_sensor,
        "target_variable": "traffic_speed (mph)",
        "timestamp_start": str(ts_start),
        "timestamp_end": str(ts_end),
        "sampling_interval": sampling_interval_str,
        "processed_shape": df_clean.shape,
        "missing_values": missing_vals,
        "duplicate_timestamps": duplicate_ts,
        "min_target": round(min_speed, 2),
        "max_target": round(max_speed, 2),
        "mean_target": round(mean_speed, 2),
        "std_target": round(std_speed, 2),
        "split_sizes": {
            "train_70pct": n_train,
            "val_15pct": n_val,
            "test_15pct": n_test,
            "total_rows": n_rows
        },
        "horizons_definition": {
            "15_minutes": "3 steps (3 x 5 min)",
            "30_minutes": "6 steps (6 x 5 min)",
            "45_minutes": "9 steps (9 x 5 min)",
            "60_minutes": "12 steps (12 x 5 min)"
        },
        "output_csv_path": output_csv
    }

if __name__ == "__main__":
    report = process_and_validate_metr_la()
    print("\n=== METR-LA PROCESSING & VALIDATION REPORT ===")
    for k, v in report.items():
        print(f"{k}: {v}")
