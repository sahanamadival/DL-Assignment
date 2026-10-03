import os
import json

notebooks = {
    "notebooks/01_dataset_analysis.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# 01. Dataset & Traffic Annotation Analysis\n", "Explores raw video traffic data, frame resolution, frame rates, and BDD100K object annotation statistics."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.data_loader import DatasetInspector\n",
                    "inspector = DatasetInspector()\n",
                    "info = inspector.inspect_raw_directory()\n",
                    "print('Dataset Summary:', info)\n"
                ]
            }
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    },
    "notebooks/02_detection_analysis.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# 02. Object Detection Analysis (YOLOv8 / YOLO11)\n", "Evaluates vehicle detection bounding boxes, confidence score distributions, and class filters."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import cv2\n",
                    "import numpy as np\n",
                    "from src.detector import YOLODetector\n",
                    "detector = YOLODetector()\n",
                    "dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)\n",
                    "res = detector.detect_frame(dummy_frame)\n",
                    "print('Detection results:', res['class_counts'])\n"
                ]
            }
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    },
    "notebooks/03_traffic_feature_extraction.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# 03. Traffic Feature Extraction & Dataset Generation\n", "Extracts vehicle counts, density, speed proxies, and flow rates, saving to `data/traffic_timeseries.csv`."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.traffic_features import TrafficFeatureExtractor\n",
                    "extractor = TrafficFeatureExtractor()\n",
                    "df = extractor.generate_extended_traffic_timeseries()\n",
                    "print(df.head())\n"
                ]
            }
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    },
    "notebooks/04_timeseries_preprocessing.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# 04. Time-Series Preprocessing & Leakage Prevention\n", "Demonstrates chronological 70/15/15 splitting and fitting scalers strictly on training data."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.preprocessing import ChronologicalPreprocessor\n",
                    "prep = ChronologicalPreprocessor()\n",
                    "df = prep.load_raw_csv()\n",
                    "tr, val, te = prep.split_chronologically(df)\n",
                    "X_tr, y_tr = prep.fit_transform_train(tr)\n",
                    "print('Train X shape:', X_tr.shape)\n"
                ]
            }
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    },
    "notebooks/05_lstm_training.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# 05. LSTM Model Training & Hyperparameter Tuning\n", "Trains 2-layer LSTM model for congestion score forecasting."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.train_lstm import LSTMTrainer\n",
                    "trainer = LSTMTrainer()\n",
                    "res = trainer.train_model()\n",
                    "print('Best val loss:', res['best_val_loss'])\n"
                ]
            }
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    },
    "notebooks/06_model_evaluation.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# 06. Model Benchmark & Comparative Evaluation\n", "Compares Naive Last-Value, Random Forest, LSTM, and GRU models across MAE, RMSE, R², and MAPE metrics."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.evaluate import Evaluator\n",
                    "evaluator = Evaluator()\n",
                    "results = evaluator.run_full_evaluation()\n",
                    "print(results)\n"
                ]
            }
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    }
}

os.makedirs("notebooks", exist_ok=True)
for nb_path, nb_content in notebooks.items():
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_content, f, indent=2)
print("Research notebooks created successfully!")
