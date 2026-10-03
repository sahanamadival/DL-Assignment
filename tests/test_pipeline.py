import os
import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.detector import YOLODetector
from src.tracker import VehicleTracker
from src.congestion import CongestionEstimator
from src.preprocessing import ChronologicalPreprocessor
from src.sequence_builder import SequenceBuilder
from src.predictor import CongestionPredictor

def test_config_loader():
    cfg = load_config()
    assert cfg.detector.model_name == "yolov8n.pt"
    assert cfg.time_series.sequence_length == 10

def test_congestion_estimator():
    estimator = CongestionEstimator()
    res = estimator.calculate_congestion(density=0.8, vehicle_count=40, relative_speed_proxy=1.5)
    assert "congestion_score" in res
    assert res["congestion_level"] in ["LOW", "MODERATE", "HIGH", "SEVERE"]

def test_preprocessing_chronological_split():
    prep = ChronologicalPreprocessor()
    df_dummy = pd.DataFrame({
        "total_vehicles": range(100),
        "cars": range(100),
        "motorcycles": [0]*100,
        "buses": [0]*100,
        "trucks": [0]*100,
        "pedestrians": [0]*100,
        "vehicle_flow": range(100),
        "density": [0.5]*100,
        "lane_occupancy": [0.5]*100,
        "average_speed": [40]*100,
        "congestion_score": [0.5]*100
    })
    tr, val, te = prep.split_chronologically(df_dummy)
    assert len(tr) == 70
    assert len(val) == 15
    assert len(te) == 15

def test_sequence_builder():
    builder = SequenceBuilder()
    X_dummy = np.random.randn(50, 10).astype(np.float32)
    y_dummy = np.random.randn(50).astype(np.float32)
    X_seq, y_seq = builder.create_sequences(X_dummy, y_dummy)
    assert X_seq.ndim == 3
    assert X_seq.shape[1] == 10

def test_predictor_inference():
    if os.path.exists("models/lstm/lstm_congestion.pt"):
        predictor = CongestionPredictor(model_type="lstm")
        dummy_hist = [{
            "total_vehicles": 20, "cars": 15, "motorcycles": 2, "buses": 2, "trucks": 1, "pedestrians": 0,
            "vehicle_flow": 30.0, "density": 0.4, "lane_occupancy": 0.5, "average_speed": 35.0, "congestion_score": 0.4
        } for _ in range(10)]
        res = predictor.predict_future_congestion(dummy_hist)
        assert "horizons" in res
        assert "15_min" in res["horizons"]
