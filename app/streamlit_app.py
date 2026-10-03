import os
import sys
import json
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.data_loader import DatasetInspector
from src.detector import YOLODetector
from src.tracker import VehicleTracker
from src.traffic_features import TrafficFeatureExtractor
from src.predictor import CongestionPredictor

# Page Setup & Styling
st.set_page_config(
    page_title="AI Traffic Monitoring & Short-Term Congestion Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design System
st.markdown("""
<style>
    /* Dark glassmorphism theme */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 5px;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-pill {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_config():
    return load_config()

@st.cache_resource
def get_models():
    detector = YOLODetector()
    tracker = VehicleTracker()
    predictor_lstm = CongestionPredictor(model_type="lstm")
    predictor_gru = CongestionPredictor(model_type="gru")
    return detector, tracker, predictor_lstm, predictor_gru

cfg = get_config()

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric-folders/100/traffic-jam.png", width=70)
st.sidebar.title("Traffic AI Control Panel")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox("Predictor Model", ["LSTM", "GRU"], index=0)
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.10, 0.90, cfg.detector.conf_threshold, 0.05)
sequence_length = st.sidebar.slider("LSTM Sequence Window Length", 5, 30, cfg.time_series.sequence_length, 1)

st.sidebar.markdown("---")
st.sidebar.info("""
**Methodology Note:**
Computer Vision (YOLO + ByteTrack) extracts frame traffic parameters.
Temporal Deep Learning (LSTM/GRU) forecasts short-term congestion score across 15m, 30m, 45m horizons.
""")

# Title Header
st.title("🚦 Real-Time Traffic Object Detection, Tracking & Congestion Prediction")
st.caption("Deep Learning Framework combining YOLOv8 / YOLO11, ByteTrack Multi-Object Tracking, and Temporal LSTM/GRU Models")
st.markdown("---")

# Navigation Tabs
tab_live, tab_stats, tab_predict, tab_benchmark = st.tabs([
    "📹 Live Video & Tracking",
    "📊 Vehicle Statistics & Analytics",
    "🔮 Future Congestion Prediction",
    "📈 Model Evaluation & Research Benchmarks"
])

# Load Models
try:
    detector, tracker, predictor_lstm, predictor_gru = get_models()
    active_predictor = predictor_lstm if selected_model == "LSTM" else predictor_gru
    models_ready = True
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    models_ready = False

# TAB 1: Live Video & Object Tracking
with tab_live:
    st.subheader("Real-Time Video Detection & Multi-Object Tracking")
    
    video_source = st.radio("Select Video Input Source:", ["Default Synthetic Sample Video", "Upload Traffic Video (MP4/AVI)"], horizontal=True)
    
    video_path = None
    if video_source == "Default Synthetic Sample Video":
        sample_path = "data/raw/sample_traffic.mp4"
        if not os.path.exists(sample_path):
            inspector = DatasetInspector()
            inspector.generate_synthetic_traffic_video(sample_path)
        video_path = sample_path
    else:
        uploaded_file = st.file_uploader("Upload Traffic Video File", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name

    if video_path and models_ready:
        col_video, col_live_metrics = st.columns([2, 1])
        
        with col_video:
            st.markdown("### Video Stream Output")
            video_placeholder = st.empty()
            
        with col_live_metrics:
            st.markdown("### Active Tracking Metrics")
            metric_active_ids = st.empty()
            metric_total_seen = st.empty()
            metric_current_speed = st.empty()
            metric_current_density = st.empty()
            metric_congestion_pill = st.empty()

        start_btn = st.button("▶ Start Real-Time Processing", type="primary")

        if start_btn:
            cap = cv2.VideoCapture(video_path)
            tracker.reset_tracker()
            frame_idx = 0
            
            history_window = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                # Run tracking frame
                res = tracker.track_frame(frame, frame_idx)
                
                # Render frame in Streamlit
                frame_rgb = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Update stats cards
                active_cnt = res["active_vehicle_count"]
                unique_cnt = res["total_unique_vehicles_seen"]
                
                # Simple real-time metrics update
                metric_active_ids.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Active Tracked Vehicles</div>
                    <div class="metric-value">{active_cnt}</div>
                </div>
                """, unsafe_allow_html=True)

                metric_total_seen.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Unique Vehicles Seen</div>
                    <div class="metric-value">{unique_cnt}</div>
                </div>
                """, unsafe_allow_html=True)

                # Compute frame density
                h_f, w_f, _ = frame.shape
                total_area = h_f * w_f
                bbox_area = sum([(obj["bbox"][2]-obj["bbox"][0])*(obj["bbox"][3]-obj["bbox"][1]) for obj in res["tracked_objects"]])
                density = round(float(bbox_area / (total_area + 1e-5)), 4)
                
                metric_current_density.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Spatial Density Ratio</div>
                    <div class="metric-value">{density:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

                # Small delay to keep UI smooth
                time.sleep(0.01)
                
                if frame_idx >= 150: # Cap live preview at 150 frames for interactive responsiveness
                    break

            cap.release()
            st.success("Video Stream Processing Window Finished!")

# TAB 2: Vehicle Statistics & Analytics
with tab_stats:
    st.subheader("Historical Traffic Time-Series Analytics")
    
    csv_path = cfg.paths.traffic_csv
    if os.path.exists(csv_path):
        df_ts = pd.read_csv(csv_path)
        
        # Determine available column mappings
        flow_col = "unique_vehicle_flow" if "unique_vehicle_flow" in df_ts.columns else ("vehicle_flow" if "vehicle_flow" in df_ts.columns else df_ts.columns[0])
        speed_col = "relative_speed_proxy" if "relative_speed_proxy" in df_ts.columns else ("average_speed" if "average_speed" in df_ts.columns else df_ts.columns[0])

        # Summary Row Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Intervals", len(df_ts))
        with col2:
            st.metric("Avg Vehicles / Interval", int(df_ts["total_vehicles"].mean()) if "total_vehicles" in df_ts.columns else 0)
        with col3:
            st.metric("Peak Unique Flow", f"{df_ts[flow_col].max():.1f}")
        with col4:
            st.metric("Avg Speed Proxy", f"{df_ts[speed_col].mean():.1f}")
        with col5:
            st.metric("Avg Congestion Score", f"{df_ts['congestion_score'].mean():.2f}" if "congestion_score" in df_ts.columns else "0.00")

        st.markdown("---")
        
        # Class Breakdown Chart
        st.markdown("### Vehicle Class Distribution")
        class_sums = {
            "Cars": int(df_ts["cars"].sum()) if "cars" in df_ts.columns else 0,
            "Motorcycles": int(df_ts["motorcycles"].sum()) if "motorcycles" in df_ts.columns else 0,
            "Buses": int(df_ts["buses"].sum()) if "buses" in df_ts.columns else 0,
            "Trucks": int(df_ts["trucks"].sum()) if "trucks" in df_ts.columns else 0,
            "Pedestrians": int(df_ts["pedestrians"].sum()) if "pedestrians" in df_ts.columns else 0
        }
        df_classes = pd.DataFrame(list(class_sums.items()), columns=["Vehicle Class", "Total Count"])
        st.bar_chart(df_classes.set_index("Vehicle Class"))

        # Time-Series Trends
        st.markdown("### Traffic Parameters Over Time")
        avail_params = [c for c in ["total_vehicles", "density", "relative_speed_proxy", "average_speed", "congestion_score", "unique_vehicle_flow", "vehicle_flow"] if c in df_ts.columns]
        if avail_params:
            chart_type = st.selectbox("Select Parameter to Plot:", avail_params)
            st.line_chart(df_ts[chart_type])

    else:
        st.warning("Traffic time-series CSV missing. Run Stage 5 feature extraction to populate.")

# TAB 3: Future Congestion Prediction
with tab_predict:
    st.subheader("🔮 Short-Term Traffic Congestion Forecasting")
    
    if os.path.exists(csv_path) and models_ready:
        df_ts = pd.read_csv(csv_path)
        
        # Select history window
        st.markdown(f"**Loaded History Input Window (Last {sequence_length} timesteps):**")
        history_window = df_ts.tail(sequence_length).to_dict(orient="records")
        display_cols = [c for c in ["video_time_seconds", "timestamp", "total_vehicles", "density", "relative_speed_proxy", "average_speed", "congestion_score"] if c in df_ts.columns]
        st.dataframe(pd.DataFrame(history_window)[display_cols])

        if st.button("🚀 Generate Short-Term Congestion Forecast", type="primary"):
            try:
                pred_res = active_predictor.predict_future_congestion(history_window)
                
                st.markdown("---")
                st.markdown(f"### Prediction Results (Model: **{pred_res['model_used']}** | Confidence: **{pred_res['prediction_confidence_pct']}%**)")
                
                col_h15, col_h30, col_h45 = st.columns(3)
                
                with col_h15:
                    h15 = pred_res["horizons"]["15_min"]
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 6px solid {h15['color_hex']};">
                        <div class="metric-label">15 Min Horizon</div>
                        <div class="metric-value">{h15['score']:.2f}</div>
                        <div class="status-pill" style="background-color: {h15['color_hex']}; color: #ffffff;">{h15['level']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_h30:
                    h30 = pred_res["horizons"]["30_min"]
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 6px solid {h30['color_hex']};">
                        <div class="metric-label">30 Min Horizon</div>
                        <div class="metric-value">{h30['score']:.2f}</div>
                        <div class="status-pill" style="background-color: {h30['color_hex']}; color: #ffffff;">{h30['level']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_h45:
                    h45 = pred_res["horizons"]["45_min"]
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 6px solid {h45['color_hex']};">
                        <div class="metric-label">45 Min Horizon</div>
                        <div class="metric-value">{h45['score']:.2f}</div>
                        <div class="status-pill" style="background-color: {h45['color_hex']}; color: #ffffff;">{h45['level']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                # Forecast timeline plot
                st.markdown("### Congestion Forecasting Timeline")
                timeline_df = pd.DataFrame({
                    "Horizon": ["Current", "15 Min", "30 Min", "45 Min"],
                    "Congestion Score": [
                        pred_res["current_congestion_score"],
                        h15["score"],
                        h30["score"],
                        h45["score"]
                    ]
                })
                st.line_chart(timeline_df.set_index("Horizon"))

            except Exception as e:
                st.error(f"Prediction Execution Error: {e}")

# TAB 4: Model Evaluation & Research Benchmarks
with tab_benchmark:
    st.subheader("📈 Research Model Evaluation & Baseline Benchmarks")
    
    metrics_path = os.path.join(cfg.paths.results_metrics_dir, "model_benchmark_metrics.json")
    fig_actual_path = os.path.join(cfg.paths.results_figures_dir, "actual_vs_predicted.png")
    fig_loss_path = os.path.join(cfg.paths.results_figures_dir, "loss_curves.png")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            benchmarks = json.load(f)

        st.markdown("### Quantitative Benchmark Comparison")
        df_bench = pd.DataFrame(benchmarks).T
        st.dataframe(df_bench.style.highlight_min(subset=["MAE", "RMSE", "MAPE_Percent"], color="#1e3a8a"))

        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            if os.path.exists(fig_actual_path):
                st.image(fig_actual_path, caption="Actual vs. Predicted Congestion Score (Test Set)", use_container_width=True)
        with col_fig2:
            if os.path.exists(fig_loss_path):
                st.image(fig_loss_path, caption="LSTM vs GRU Training & Validation Loss Curves", use_container_width=True)
    else:
        st.info("No benchmark results found. Run `python -m src.evaluate` to generate evaluation plots.")
