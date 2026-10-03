import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from src.config import load_config
from src.tracker import VehicleTracker
from src.congestion import CongestionEstimator

class TrafficFeatureExtractor:
    """Extracts real traffic measurements frame-by-frame from video using YOLO and ByteTrack."""

    def __init__(self, tracker: Optional[VehicleTracker] = None):
        self.cfg = load_config()
        self.tracker = tracker or VehicleTracker()
        self.estimator = CongestionEstimator()

    def extract_from_real_video(self, video_path: str = "data/raw/videos/test_traffic.mp4", aggregation_interval_seconds: float = 1.0, output_csv: str = "data/traffic_timeseries.csv") -> Dict[str, Any]:
        """Processes real video, runs ByteTrack, aggregates features per interval, saves CSV, plots, and validates data."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Real traffic video missing at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frame_area = frame_width * frame_height

        self.tracker.reset_tracker()
        
        frames_per_interval = max(1, int(round(aggregation_interval_seconds * native_fps)))
        
        frame_idx = 0
        interval_data_buffer = []
        time_series_records = []

        print(f"[INFO] Processing real video '{video_path}'...")
        print(f"[INFO] Native Resolution: {frame_width}x{height_f if 'height_f' in locals() else frame_height} @ {native_fps} FPS")
        print(f"[INFO] Aggregation Interval: {aggregation_interval_seconds} sec ({frames_per_interval} frames/interval)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            track_res = self.tracker.track_frame(frame, frame_idx=frame_idx)
            interval_data_buffer.append(track_res)

            if frame_idx % frames_per_interval == 0:
                interval_idx = len(time_series_records) + 1
                video_time_sec = round(interval_idx * aggregation_interval_seconds, 2)
                rec = self._aggregate_interval(interval_data_buffer, video_time_sec, total_frame_area)
                time_series_records.append(rec)
                interval_data_buffer = []

        cap.release()

        # Handle leftover frames in final interval
        if len(interval_data_buffer) > 0:
            interval_idx = len(time_series_records) + 1
            video_time_sec = round(frame_idx / native_fps, 2)
            rec = self._aggregate_interval(interval_data_buffer, video_time_sec, total_frame_area)
            time_series_records.append(rec)

        df = pd.DataFrame(time_series_records)

        # Save to data/traffic_timeseries.csv
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Real traffic feature time-series saved to: {output_csv} ({len(df)} rows)")

        # Data Quality Checks
        quality_report = self._run_data_quality_checks(df)

        # Generate Real Plots
        fig_paths = self.generate_real_plots(df)

        # Critical Forecasting Suitability Check
        total_duration_sec = round(frame_idx / native_fps, 2)
        forecasting_suitability = self._evaluate_forecasting_suitability(total_duration_sec, len(df), aggregation_interval_seconds)

        return {
            "video_path": video_path,
            "video_duration_seconds": total_duration_sec,
            "frames_processed": frame_idx,
            "native_fps": round(native_fps, 2),
            "aggregation_interval_seconds": aggregation_interval_seconds,
            "num_rows": len(df),
            "csv_path": output_csv,
            "columns": list(df.columns),
            "quality_report": quality_report,
            "generated_plots": fig_paths,
            "forecasting_suitability": forecasting_suitability,
            "head_rows": df.head(10).to_dict(orient="records")
        }

    def _aggregate_interval(self, buffer: List[Dict[str, Any]], video_time_sec: float, frame_area: float) -> Dict[str, Any]:
        """Aggregates buffer of frame tracking dictionaries into a single time-series row."""
        if not buffer:
            return {
                "video_time_seconds": video_time_sec,
                "total_vehicles": 0, "cars": 0, "motorcycles": 0, "buses": 0, "trucks": 0, "pedestrians": 0,
                "active_vehicle_count": 0, "unique_vehicle_flow": 0,
                "density": 0.0, "road_region_occupancy": float("nan"),
                "relative_speed_proxy": 0.0, "congestion_score": 0.0, "congestion_level": "LOW"
            }

        # Average counts per frame across the interval
        avg_cars = np.mean([f["class_counts"].get("car", 0) for f in buffer])
        avg_buses = np.mean([f["class_counts"].get("bus", 0) for f in buffer])
        avg_trucks = np.mean([f["class_counts"].get("truck", 0) for f in buffer])
        avg_motorcycles = np.mean([f["class_counts"].get("motorcycle", 0) for f in buffer])
        avg_pedestrians = np.mean([f["class_counts"].get("person", 0) for f in buffer])
        
        # Active tracked vehicles (excluding pedestrians)
        avg_active_vehicles = np.mean([f["active_vehicle_count"] for f in buffer])

        # Unique vehicle flow = unique IDs seen in this interval
        interval_unique_ids = set()
        for f in buffer:
            for obj in f["tracked_objects"]:
                if obj["class_name"] in ["car", "bus", "truck", "motorcycle", "bicycle"]:
                    interval_unique_ids.add(obj["track_id"])
        unique_flow = len(interval_unique_ids)

        # Image-space occupancy density proxy
        frame_densities = []
        frame_speeds = []
        for f in buffer:
            total_bbox_area = sum([(obj["bbox"][2]-obj["bbox"][0]) * (obj["bbox"][3]-obj["bbox"][1]) for obj in f["tracked_objects"]])
            frame_densities.append(total_bbox_area / (frame_area + 1e-5))

            speeds = [obj["relative_speed_proxy_px_per_frame"] for obj in f["tracked_objects"] if obj["relative_speed_proxy_px_per_frame"] > 0]
            if speeds:
                frame_speeds.append(np.mean(speeds))

        density = round(float(np.clip(np.mean(frame_densities) if frame_densities else 0.0, 0.0, 1.0)), 4)
        speed_proxy = round(float(np.mean(frame_speeds) if frame_speeds else 0.0), 2)

        # Congestion score estimation
        cong_res = self.estimator.calculate_congestion(
            density=density,
            vehicle_count=int(round(avg_active_vehicles)),
            relative_speed_proxy=speed_proxy
        )

        return {
            "video_time_seconds": video_time_sec,
            "total_vehicles": int(round(avg_active_vehicles)),
            "cars": int(round(avg_cars)),
            "motorcycles": int(round(avg_motorcycles)),
            "buses": int(round(avg_buses)),
            "trucks": int(round(avg_trucks)),
            "pedestrians": int(round(avg_pedestrians)),
            "active_vehicle_count": int(round(avg_active_vehicles)),
            "unique_vehicle_flow": unique_flow,
            "density": density,
            "road_region_occupancy": float("nan"),  # Uncalibrated lane geometry
            "relative_speed_proxy": speed_proxy,
            "congestion_score": cong_res["congestion_score"],
            "congestion_level": cong_res["congestion_level"]
        }

    def _run_data_quality_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs quality validation checks on extracted real dataset."""
        numeric_cols = ["total_vehicles", "cars", "motorcycles", "buses", "trucks", "density", "relative_speed_proxy", "congestion_score"]
        
        missing_count = int(df[numeric_cols].isnull().sum().sum())
        negative_count = int((df[numeric_cols] < 0).sum().sum())
        density_out_of_bounds = int(((df["density"] < 0) | (df["density"] > 1.0)).sum())
        duplicate_times = int(df["video_time_seconds"].duplicated().sum())

        return {
            "missing_values_count": missing_count,
            "negative_values_count": negative_count,
            "density_out_of_bounds_count": density_out_of_bounds,
            "duplicate_timestamps_count": duplicate_times,
            "is_clean": (missing_count == 0 and negative_count == 0 and density_out_of_bounds == 0 and duplicate_times == 0)
        }

    def generate_real_plots(self, df: pd.DataFrame, fig_dir: str = "results/figures") -> Dict[str, str]:
        """Generates 4 plots from real extracted data and saves to results/figures/."""
        os.makedirs(fig_dir, exist_ok=True)
        paths = {}

        # 1. Real Vehicle Count
        plt.figure(figsize=(9, 4.5))
        plt.plot(df["video_time_seconds"], df["total_vehicles"], marker="o", color="#2563eb", linewidth=2.0)
        plt.title("Real Vehicle Count Over Video Time", fontsize=12, fontweight="bold")
        plt.xlabel("Video Time (seconds)")
        plt.ylabel("Active Vehicles")
        plt.grid(True, linestyle=":", alpha=0.6)
        p1 = os.path.join(fig_dir, "real_vehicle_count.png")
        plt.savefig(p1, dpi=300, bbox_inches="tight")
        plt.close()
        paths["real_vehicle_count"] = p1

        # 2. Real Density
        plt.figure(figsize=(9, 4.5))
        plt.plot(df["video_time_seconds"], df["density"], marker="s", color="#d97706", linewidth=2.0)
        plt.title("Image-Space Density Proxy Over Video Time", fontsize=12, fontweight="bold")
        plt.xlabel("Video Time (seconds)")
        plt.ylabel("Density Ratio [0 - 1]")
        plt.grid(True, linestyle=":", alpha=0.6)
        p2 = os.path.join(fig_dir, "real_density.png")
        plt.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close()
        paths["real_density"] = p2

        # 3. Real Relative Speed Proxy
        plt.figure(figsize=(9, 4.5))
        plt.plot(df["video_time_seconds"], df["relative_speed_proxy"], marker="^", color="#059669", linewidth=2.0)
        plt.title("Relative Speed Proxy Over Video Time", fontsize=12, fontweight="bold")
        plt.xlabel("Video Time (seconds)")
        plt.ylabel("Speed Proxy (pixels/frame)")
        plt.grid(True, linestyle=":", alpha=0.6)
        p3 = os.path.join(fig_dir, "real_speed_proxy.png")
        plt.savefig(p3, dpi=300, bbox_inches="tight")
        plt.close()
        paths["real_speed_proxy"] = p3

        # 4. Real Congestion Score
        plt.figure(figsize=(9, 4.5))
        plt.plot(df["video_time_seconds"], df["congestion_score"], marker="d", color="#dc2626", linewidth=2.0)
        plt.title("Real Congestion Score Over Video Time", fontsize=12, fontweight="bold")
        plt.xlabel("Video Time (seconds)")
        plt.ylabel("Congestion Score [0 - 1]")
        plt.grid(True, linestyle=":", alpha=0.6)
        p4 = os.path.join(fig_dir, "real_congestion_score.png")
        plt.savefig(p4, dpi=300, bbox_inches="tight")
        plt.close()
        paths["real_congestion_score"] = p4

        return paths

    def _evaluate_forecasting_suitability(self, duration_sec: float, num_timesteps: int, interval_sec: float) -> Dict[str, Any]:
        """Evaluates whether the extracted time-series duration is sufficient for long-horizon forecasting."""
        is_sufficient_for_forecasting = (duration_sec >= 1800.0) # Requires at least 30 minutes of video for 15-30m forecasting
        
        status_text = "Sufficient for forecasting experiment" if is_sufficient_for_forecasting else "Suitable for CV/feature-extraction testing"
        reason = (
            "Dataset has sufficient temporal coverage (>= 30 mins) for future congestion forecasting."
            if is_sufficient_for_forecasting else
            f"The input video duration ({duration_sec}s / {num_timesteps} steps) is suitable for CV feature-extraction pipeline verification, but insufficient by itself for long-horizon 15-30-45 minute forecasting."
        )

        return {
            "video_duration_seconds": duration_sec,
            "total_time_steps": num_timesteps,
            "sampling_interval_seconds": interval_sec,
            "status": status_text,
            "is_sufficient_for_forecasting": is_sufficient_for_forecasting,
            "explanation": reason
        }

if __name__ == "__main__":
    extractor = TrafficFeatureExtractor()
    res = extractor.extract_from_real_video("data/raw/videos/test_traffic.mp4", aggregation_interval_seconds=1.0)
    print("\n=== REAL TRAFFIC FEATURE EXTRACTION COMPLETED ===")
    print(f"Video Duration: {res['video_duration_seconds']} sec ({res['frames_processed']} frames)")
    print(f"Time-Series Rows Generated: {res['num_rows']}")
    print(f"Forecasting Suitability: {res['forecasting_suitability']['status']}")
    print("\nFirst 5 Rows:")
    print(pd.DataFrame(res['head_rows']).head(5))
