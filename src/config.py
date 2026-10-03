import os
import yaml
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class DetectorConfig(BaseModel):
    model_name: str = "yolov8n.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cpu"
    target_classes: Dict[int, str] = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic light",
        11: "stop sign"
    }

class TrackerConfig(BaseModel):
    type: str = "bytetrack"
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30

class SpeedEstimationConfig(BaseModel):
    pixel_to_meter_scale: float = 0.05
    fps: float = 30.0

class LaneOccupancyConfig(BaseModel):
    enabled: bool = True
    default_lane_area_ratio: float = 0.85

class CongestionThresholdsConfig(BaseModel):
    low_max: float = 0.30
    moderate_max: float = 0.60
    high_max: float = 0.85

class TrafficAnalysisConfig(BaseModel):
    time_interval_seconds: int = 60
    speed_estimation: SpeedEstimationConfig = SpeedEstimationConfig()
    lane_occupancy: LaneOccupancyConfig = LaneOccupancyConfig()
    congestion_thresholds: CongestionThresholdsConfig = CongestionThresholdsConfig()

class TimeSeriesConfig(BaseModel):
    sequence_length: int = 10
    prediction_horizon: int = 15
    feature_columns: List[str] = [
        "total_vehicles", "cars", "motorcycles", "buses", "trucks", 
        "pedestrians", "vehicle_flow", "density", "lane_occupancy", "average_speed"
    ]
    target_column: str = "congestion_score"

class ModelTrainingConfig(BaseModel):
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    dropout: float = 0.2
    patience: int = 10

class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    traffic_csv: str = "data/traffic_timeseries.csv"
    detector_model_dir: str = "models/detector"
    tracker_model_dir: str = "models/tracker"
    lstm_model_dir: str = "models/lstm"
    gru_model_dir: str = "models/gru"
    results_figures_dir: str = "results/figures"
    results_metrics_dir: str = "results/metrics"
    results_predictions_dir: str = "results/predictions"

class AppConfig(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    traffic_analysis: TrafficAnalysisConfig = TrafficAnalysisConfig()
    time_series: TimeSeriesConfig = TimeSeriesConfig()
    model_training: ModelTrainingConfig = ModelTrainingConfig()
    paths: PathsConfig = PathsConfig()

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Loads configuration from YAML file and validates via Pydantic schema."""
    if not os.path.exists(config_path):
        return AppConfig()
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    
    return AppConfig(**raw_data)
