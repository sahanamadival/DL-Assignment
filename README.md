# Real-Time Traffic Object Detection, Tracking and Short-Term Congestion Prediction Using YOLO and LSTM

An academic research-grade, full-stack Deep Learning framework combining real-time computer vision object detection (YOLOv8 / YOLO11), multi-object vehicle tracking (ByteTrack), traffic parameter extraction, and temporal deep learning models (LSTM & GRU) for short-term traffic forecasting.

---

## 1. Project Architecture & Dataset Separation

To maintain research integrity, the framework strictly decouples Computer Vision feature extraction from Time-Series Temporal Forecasting:

### A. Computer Vision Dataset (`data/traffic_timeseries.csv`)
* **Pipeline**: Real Traffic Video (`data/raw/videos/test_traffic.mp4`) $\rightarrow$ YOLO Detector $\rightarrow$ ByteTrack Tracking $\rightarrow$ Real Feature Extractor.
* **Purpose**: Real-time vehicle counting, class breakdown, spatial density proxy, relative speed proxy ($\text{pixels/frame}$), and current heuristic congestion level (`LOW`, `MODERATE`, `HIGH`, `SEVERE`).

### B. Time-Series Forecasting Dataset (`data/processed/metr_la_forecasting.csv`)
* **Pipeline**: METR-LA Real Traffic Sensor Dataset $\rightarrow$ Chronological Preprocessing $\rightarrow$ Leakage-Free Sequence Builder $\rightarrow$ Baseline Models & LSTM/GRU Models.
* **Purpose**: Multi-horizon temporal traffic speed forecasting ($15\text{m}, 30\text{m}, 45\text{m}$ ahead).

```
Real Traffic Video (.mp4)                           METR-LA Sensor Dataset (METR-LA.csv)
        │                                                           │
        ▼                                                           ▼
YOLOv8 / YOLO11 Detection                                Chronological 70/15/15 Split
        │                                                           │
        ▼                                                           ▼
ByteTrack Vehicle Tracking                                MinMaxScaler (Fitted ON TRAIN ONLY)
        │                                                           │
        ▼                                                           ▼
Real Feature Extraction                                  Sliding Window Sequences (seq_15min, seq_30min, seq_45min)
        │                                                           │
        ▼                                                           ▼
data/traffic_timeseries.csv                              data/processed/sequences/*.npz
                                                                    │
                                                                    ▼
                                                         Baseline Models (Naive & Random Forest)
```

---

## 2. METR-LA Forecasting Dataset Preprocessing & Sequences

* **Sensor Selected**: Sensor ID `'773869'` (Continuous speed measurement in $\text{mph}$)
* **Total Observations**: `34,272` 5-minute time steps ($\approx 4\text{ months}$, `2012-03-01` $\rightarrow$ `2012-06-27`)
* **Sampling Interval**: Exactly `5 minutes` (`00:05:00`)
* **Input Window Length**: `sequence_length = 12` ($12 \times 5\text{ min} = 60\text{ minutes}$ history)
* **Forecast Horizons**:
  * **15 Minutes**: `3 time steps` ($t+3$)
  * **30 Minutes**: `6 time steps` ($t+6$)
  * **45 Minutes**: `9 time steps` ($t+9$)

---

## 3. Stage 5 — Baseline Model Results (METR-LA Test Set)

Quantitative evaluation of **Naive Persistence** and **Random Forest Regressor** baselines on the unseen METR-LA test set ($5,128$ to $5,122$ test sequences):

| Model | Horizon | MAE (mph) ↓ | RMSE (mph) ↓ | $R^2$ Score ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Persistence** | 15 min | 3.2745 | 9.3848 | 0.8226 |
| **Random Forest** | 15 min | 4.0581 | 9.1959 | 0.8296 |
| **Naive Persistence** | 30 min | 4.6380 | 12.5121 | 0.6848 |
| **Random Forest** | 30 min | 6.0015 | 11.9656 | 0.7117 |
| **Naive Persistence** | 45 min | 5.9205 | 14.9671 | 0.5491 |
| **Random Forest** | 45 min | 7.5599 | 13.8756 | 0.6125 |

*Metrics recorded in [`results/metrics/baseline_results.csv`](file:///c:/Users/Sahana%20Madival/OneDrive/Attachments/Documents/DL/results/metrics/baseline_results.csv) and plots saved in [`results/figures/`](file:///c:/Users/Sahana%20Madival/OneDrive/Attachments/Documents/DL/results/figures/).*

---

## 4. Real Traffic Feature Schema (`data/traffic_timeseries.csv`)

| Feature Name | Description |
| :--- | :--- |
| `video_time_seconds` | Video elapsed time index ($\text{seconds}$) |
| `total_vehicles` | Average active tracked vehicle count in interval |
| `cars` | Car class count |
| `motorcycles` | Motorcycle count |
| `buses` | Bus class count |
| `trucks` | Heavy vehicle / truck count |
| `pedestrians` | Pedestrian count |
| `active_vehicle_count` | Active vehicle count (excluding pedestrians) |
| `unique_vehicle_flow` | Count of unique ByteTrack IDs observed in interval |
| `density` | Image-space bounding box occupancy density proxy $[0.0 - 1.0]$ |
| `road_region_occupancy` | `NaN` (Uncalibrated lane geometry) |
| `relative_speed_proxy` | Average trajectory displacement ($\text{pixels/frame}$) |
| `congestion_score` | Project-defined empirical heuristic score $[0.0 - 1.0]$ |
| `congestion_level` | Categorical level (`LOW`, `MODERATE`, `HIGH`, `SEVERE`) |

---

## 5. Project Directory Structure

```
traffic-congestion-ai/
├── data/
│   ├── raw/
│   │   ├── videos/ (test_traffic.mp4)
│   │   └── forecasting/ (METR-LA.csv)
│   ├── processed/
│   │   ├── metr_la_forecasting.csv
│   │   └── sequences/ (seq_15min.npz, seq_30min.npz, seq_45min.npz)
│   └── traffic_timeseries.csv (Extracted real video features)
├── models/
│   ├── scaler.pkl (Fitted on Train split only)
│   ├── detector/
│   ├── tracker/
│   ├── lstm/
│   └── gru/
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_detection_analysis.ipynb
│   ├── 03_traffic_feature_extraction.ipynb
│   ├── 04_timeseries_preprocessing.ipynb
│   ├── 05_lstm_training.ipynb
│   └── 06_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── metr_la_preprocessor.py (METR-LA preprocessor & sequence builder)
│   ├── detector.py
│   ├── tracker.py
│   ├── traffic_features.py (Real feature extractor)
│   ├── congestion.py
│   ├── synthetic_data.py (Isolated demo fallback)
│   ├── sequence_builder.py
│   ├── train_baselines.py (Naive & Random Forest baseline runner)
│   ├── train_lstm.py
│   ├── train_gru.py
│   ├── evaluate.py
│   └── predictor.py
├── backend/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── results/
│   ├── figures/ (baseline_15min.png, baseline_30min.png, baseline_45min.png, baseline_metrics.png)
│   ├── metrics/ (baseline_results.csv)
│   └── predictions/ (naive_*_predictions.csv, rf_*_predictions.csv)
├── tests/
│   └── test_pipeline.py
├── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Execution Instructions

### 1. Run Real Video Detection & ByteTrack Verification
```bash
python -m src.detector --video data/raw/videos/test_traffic.mp4
python -m src.tracker --video data/raw/videos/test_traffic.mp4
python -m src.traffic_features
```

### 2. Preprocess METR-LA Forecasting Dataset & Build Sequences
```bash
python -m src.metr_la_preprocessor
```

### 3. Run Baseline Model Training & Evaluation
```bash
python -m src.train_baselines
```

### 4. Run PyTorch Deep Learning Model Training (LSTM & GRU)
```bash
python -m src.train_lstm
python -m src.train_gru
```

### 5. Run Final Model Evaluation & Benchmark Comparison
```bash
python -m src.evaluate_final
```

### 6. Run Automated Tests
```bash
pytest tests/
```

---

## 7. Stage 7 — Final Model Benchmark Evaluation & Experimental Methodology

### Experimental Methodology
* **Dataset**: Real METR-LA Traffic Speed Sensor Dataset (`sensor_id: 773869`)
* **Split Strategy**: Strict chronological $70\%$ train ($23,990$ steps), $15\%$ validation ($5,140$ steps), $15\%$ test ($5,142$ steps).
* **Scaling**: MinMaxScaler fitted **strictly on the Training split only** to prevent data leakage.
* **Input Window**: $60\text{ minutes}$ history ($12$ time steps @ $5\text{-min}$ sampling interval).
* **Forecasting Horizons**: $15\text{-min}$ ($t+3$), $30\text{-min}$ ($t+6$), and $45\text{-min}$ ($t+9$).
* **Models Evaluated**: Naive Persistence, Random Forest Regressor, PyTorch LSTM, and PyTorch GRU.
* **Evaluation Protocol**: Inverse-transformed predictions evaluated on unseen METR-LA test sequences ($5,128$ to $5,122$ test samples).

### Final Benchmark Comparison Table

| Model | Horizon | MAE (mph) ↓ | RMSE (mph) ↓ | $R^2$ Score ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Persistence** | **15 min** | 3.2745 | 9.3848 | 0.8226 |
| **Random Forest** | **15 min** | 4.0581 | 9.1959 | 0.8296 |
| **PyTorch LSTM** | **15 min** | 4.7106 | 9.1799 | 0.8302 |
| **PyTorch GRU** | **15 min** | **3.9899** | **9.0095** | **0.8365** |
| **Naive Persistence** | **30 min** | **4.6380** | 12.5121 | 0.6848 |
| **Random Forest** | **30 min** | 6.0015 | **11.9656** | **0.7117** |
| **PyTorch LSTM** | **30 min** | 9.0255 | 12.6283 | 0.6789 |
| **PyTorch GRU** | **30 min** | 7.3849 | 12.0094 | 0.7096 |
| **Naive Persistence** | **45 min** | **5.9205** | 14.9671 | 0.5491 |
| **Random Forest** | **45 min** | 7.5599 | 13.8756 | 0.6125 |
| **PyTorch LSTM** | **45 min** | 8.1553 | **13.8140** | **0.6159** |
| **PyTorch GRU** | **45 min** | 8.5627 | 13.8584 | 0.6135 |

*Metrics saved in [`results/metrics/final_model_comparison.csv`](file:///c:/Users/Sahana%20Madival/OneDrive/Attachments/Documents/DL/results/metrics/final_model_comparison.csv).*

