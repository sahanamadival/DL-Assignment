import os
import cv2
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import load_config
from src.detector import YOLODetector
from src.tracker import VehicleTracker
from src.traffic_features import TrafficFeatureExtractor
from src.predictor import CongestionPredictor

# Initialize FastAPI App
app = FastAPI(
    title="Real-Time Traffic Object Detection, Tracking & Congestion Prediction API",
    description="Research-grade REST API combining YOLOv8/YOLO11 object detection, ByteTrack tracking, traffic analytics, and LSTM/GRU short-term congestion forecasting.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model singletons
cfg = load_config()
detector_instance: Optional[YOLODetector] = None
tracker_instance: Optional[VehicleTracker] = None
predictor_lstm: Optional[CongestionPredictor] = None
predictor_gru: Optional[CongestionPredictor] = None

@app.on_event("startup")
def startup_event():
    global detector_instance, tracker_instance, predictor_lstm, predictor_gru
    print("[INFO] Initializing FastAPI Backend Models...")
    try:
        detector_instance = YOLODetector()
        tracker_instance = VehicleTracker()
        predictor_lstm = CongestionPredictor(model_type="lstm")
        predictor_gru = CongestionPredictor(model_type="gru")
        print("[INFO] All models loaded successfully!")
    except Exception as e:
        print(f"[WARNING] Startup model loading warning: {e}")

# Pydantic Request/Response Schemas
class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    config: Dict[str, Any]

class DetectionItem(BaseModel):
    bbox: List[float]
    class_id: int
    class_name: str
    confidence: float

class DetectResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    total_detections: int
    class_counts: Dict[str, int]
    detections: List[DetectionItem]

class PredictRequestItem(BaseModel):
    total_vehicles: int
    cars: int
    motorcycles: int
    buses: int
    trucks: int
    pedestrians: int
    vehicle_flow: float
    density: float
    lane_occupancy: float
    average_speed: float
    congestion_score: Optional[float] = 0.5

class PredictRequest(BaseModel):
    model_type: str = Field(default="lstm", description="Choose 'lstm' or 'gru'")
    history: List[PredictRequestItem]

class HorizonPrediction(BaseModel):
    score: float
    level: str
    color_hex: str

class PredictResponse(BaseModel):
    model_used: str
    sequence_length_used: int
    prediction_confidence_pct: float
    current_congestion_score: float
    horizons: Dict[str, HorizonPrediction]

# API Endpoints
@app.get("/", summary="Root Endpoint")
def read_root():
    return {
        "title": "Traffic Congestion Prediction API",
        "status": "online",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "yolo_detector": detector_instance is not None,
            "bytetrack_tracker": tracker_instance is not None,
            "lstm_predictor": predictor_lstm is not None,
            "gru_predictor": predictor_gru is not None
        },
        "config": {
            "model_name": cfg.detector.model_name,
            "sequence_length": cfg.time_series.sequence_length,
            "prediction_horizon": cfg.time_series.prediction_horizon
        }
    }

@app.post("/detect", response_model=DetectResponse, summary="Run Object Detection on Image")
async def detect_image(file: UploadFile = File(...), conf_threshold: float = Query(0.35, ge=0.1, le=1.0)):
    if detector_instance is None:
        raise HTTPException(status_code=500, detail="YOLO Detector not initialized.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    h, w, _ = image.shape
    res = detector_instance.detect_frame(image, conf_threshold=conf_threshold)

    return {
        "filename": file.filename or "uploaded_image.jpg",
        "image_width": w,
        "image_height": h,
        "total_detections": res["total_count"],
        "class_counts": res["class_counts"],
        "detections": res["detections"]
    }

@app.post("/process-video", summary="Process Video with YOLO + ByteTrack & Extract Traffic Features")
async def process_video(file: UploadFile = File(...), interval_frames: int = Query(30, ge=5, le=150)):
    # Save uploaded video to temporary file
    suffix = os.path.splitext(file.filename or ".mp4")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        extractor = TrafficFeatureExtractor()
        df_features = extractor.extract_from_video(tmp_path, interval_frames=interval_frames)
        records = df_features.to_dict(orient="records")
        return {
            "filename": file.filename,
            "total_interval_samples": len(records),
            "extracted_features": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/predict", response_model=PredictResponse, summary="Forecast Short-Term Future Traffic Congestion")
def predict_congestion(req: PredictRequest):
    if len(req.history) < cfg.time_series.sequence_length:
        raise HTTPException(
            status_code=400, 
            detail=f"History length must be at least {cfg.time_series.sequence_length}, got {len(req.history)}."
        )

    model_choice = req.model_type.lower()
    predictor = predictor_lstm if model_choice == "lstm" else predictor_gru
    
    if predictor is None:
        raise HTTPException(status_code=500, detail=f"Predictor model '{model_choice}' is not available.")

    try:
        history_dicts = [item.dict() for item in req.history]
        result = predictor.predict_future_congestion(history_dicts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/traffic-analysis", summary="Compute Current Traffic Analytics Summary")
def traffic_analysis(history: List[PredictRequestItem]):
    if not history:
        raise HTTPException(status_code=400, detail="Empty history list.")

    df_h = pd.DataFrame([item.dict() for item in history])
    
    summary = {
        "sample_count": len(df_h),
        "avg_total_vehicles": round(float(df_h["total_vehicles"].mean()), 2),
        "peak_vehicles": int(df_h["total_vehicles"].max()),
        "avg_speed_kmh": round(float(df_h["average_speed"].mean()), 2),
        "avg_density": round(float(df_h["density"].mean()), 4),
        "avg_congestion_score": round(float(df_h["congestion_score"].mean()), 4),
        "vehicle_type_breakdown": {
            "cars": int(df_h["cars"].sum()),
            "motorcycles": int(df_h["motorcycles"].sum()),
            "buses": int(df_h["buses"].sum()),
            "trucks": int(df_h["trucks"].sum()),
            "pedestrians": int(df_h["pedestrians"].sum())
        }
    }
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
