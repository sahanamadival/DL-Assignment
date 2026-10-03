import os
import numpy as np
import pandas as pd
from src.config import load_config
from src.congestion import CongestionEstimator

def generate_synthetic_demo_timeseries(output_csv: str = "data/processed/synthetic_demo_timeseries.csv", num_samples: int = 600) -> pd.DataFrame:
    """DEMO / TEST DATA ONLY.
    Generates synthetic time-series for offline code testing.
    DO NOT use as real experimental dataset.
    """
    cfg = load_config()
    estimator = CongestionEstimator()
    np.random.seed(42)
    records = []
    base_time = pd.Timestamp("2026-09-23 07:00:00")

    for t in range(num_samples):
        current_time = base_time + pd.Timedelta(minutes=t)
        hour = current_time.hour + current_time.minute / 60.0

        morning_peak = np.exp(-((hour - 8.0) ** 2) / 1.5)
        evening_peak = np.exp(-((hour - 17.5) ** 2) / 2.0)
        base_intensity = 0.20 + 0.65 * morning_peak + 0.75 * evening_peak + np.random.normal(0, 0.05)
        intensity = float(np.clip(base_intensity, 0.05, 0.98))

        total_vehicles = int(round(10 + intensity * 65 + np.random.randint(-3, 4)))
        cars = int(round(total_vehicles * (0.65 + np.random.uniform(-0.05, 0.05))))
        motorcycles = int(round(total_vehicles * (0.15 + np.random.uniform(-0.03, 0.03))))
        buses = int(round(total_vehicles * (0.10 + np.random.uniform(-0.02, 0.02))))
        trucks = int(round(total_vehicles * (0.07 + np.random.uniform(-0.02, 0.02))))
        pedestrians = max(0, total_vehicles - (cars + motorcycles + buses + trucks))

        density = round(float(np.clip(intensity * 0.85 + np.random.normal(0, 0.02), 0.05, 0.98)), 4)
        speed_proxy = round(float(max(1.0, 15.0 - (density * 12.0) + np.random.normal(0, 0.5))), 2)

        cong_res = estimator.calculate_congestion(
            density=density,
            vehicle_count=total_vehicles,
            relative_speed_proxy=speed_proxy
        )

        records.append({
            "video_time_seconds": t * 60,
            "total_vehicles": total_vehicles,
            "cars": cars,
            "motorcycles": motorcycles,
            "buses": buses,
            "trucks": trucks,
            "pedestrians": pedestrians,
            "active_vehicle_count": total_vehicles,
            "unique_vehicle_flow": total_vehicles,
            "density": density,
            "road_region_occupancy": float("nan"),
            "relative_speed_proxy": speed_proxy,
            "congestion_score": cong_res["congestion_score"],
            "congestion_level": cong_res["congestion_level"]
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[DEMO DATA ONLY] Saved synthetic dataset to {output_csv}")
    return df

if __name__ == "__main__":
    generate_synthetic_demo_timeseries()
