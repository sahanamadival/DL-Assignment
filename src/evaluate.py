import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import load_config
from src.preprocessing import ChronologicalPreprocessor
from src.sequence_builder import SequenceBuilder
from src.train_lstm import LSTMModel, LSTMTrainer
from src.train_gru import GRUModel, GRUTrainer

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error (handling zeros)."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

class Evaluator:
    """Evaluates Baseline, LSTM, and GRU models on the test split."""

    def __init__(self):
        self.cfg = load_config()
        self.prep = ChronologicalPreprocessor()
        self.builder = SequenceBuilder()
        self.fig_dir = self.cfg.paths.results_figures_dir
        self.metrics_dir = self.cfg.paths.results_metrics_dir
        
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Trains models (if needed), evaluates on Test set, saves metrics and plots."""
        # 1. Load Data & Create Sequences
        df = self.prep.load_raw_csv()
        train_df, val_df, test_df = self.prep.split_chronologically(df)

        X_tr_s, y_tr_s = self.prep.fit_transform_train(train_df)
        X_val_s, y_val_s = self.prep.transform_eval(val_df)
        X_te_s, y_te_s = self.prep.transform_eval(test_df)

        X_tr_seq, y_tr_seq = self.builder.create_sequences(X_tr_s, y_tr_s)
        X_val_seq, y_val_seq = self.builder.create_sequences(X_val_s, y_val_s)
        X_te_seq, y_te_seq = self.builder.create_sequences(X_te_s, y_te_s)

        # Rescale y_test back to original scale [0, 1]
        y_test_orig = self.prep.target_scaler.inverse_transform(y_te_seq.reshape(-1, 1)).flatten()

        results = {}

        # 2. Baseline Model 1: Naive Persistence (Last Value)
        # Predict y[t] = y_seq[-1] from sequence
        y_pred_naive_s = X_te_seq[:, -1, -1]  # using target position or last feature
        y_pred_naive = self.prep.target_scaler.inverse_transform(y_pred_naive_s.reshape(-1, 1)).flatten()
        results["Naive_Last_Value"] = self._compute_metrics(y_test_orig, y_pred_naive)

        # 3. Baseline Model 2: Random Forest Regressor
        # Flatten 3D sequences (num_samples, seq_len * num_features)
        X_tr_flat = X_tr_seq.reshape(X_tr_seq.shape[0], -1)
        X_te_flat = X_te_seq.reshape(X_te_seq.shape[0], -1)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_tr_flat, y_tr_seq)
        y_pred_rf_s = rf.predict(X_te_flat)
        y_pred_rf = self.prep.target_scaler.inverse_transform(y_pred_rf_s.reshape(-1, 1)).flatten()
        results["Random_Forest"] = self._compute_metrics(y_test_orig, y_pred_rf)

        # 4. Deep Learning Model: LSTM
        lstm_trainer = LSTMTrainer()
        lstm_res = lstm_trainer.train_model()
        lstm_checkpoint = torch.load(lstm_res["model_path"])
        
        device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.detector.device == "cuda" else "cpu")
        lstm_model = LSTMModel(
            input_dim=lstm_checkpoint["input_dim"],
            hidden_dim_1=lstm_checkpoint["hidden_dim_1"],
            hidden_dim_2=lstm_checkpoint["hidden_dim_2"],
            dropout=lstm_checkpoint["dropout"]
        ).to(device)
        lstm_model.load_state_dict(lstm_checkpoint["model_state"])
        lstm_model.eval()

        with torch.no_grad():
            tx_tensor = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
            y_pred_lstm_s = lstm_model(tx_tensor).cpu().numpy()

        y_pred_lstm = self.prep.target_scaler.inverse_transform(y_pred_lstm_s.reshape(-1, 1)).flatten()
        results["LSTM"] = self._compute_metrics(y_test_orig, y_pred_lstm)

        # 5. Deep Learning Model: GRU
        gru_trainer = GRUTrainer()
        gru_res = gru_trainer.train_model()
        gru_checkpoint = torch.load(gru_res["model_path"])

        gru_model = GRUModel(
            input_dim=gru_checkpoint["input_dim"],
            hidden_dim_1=gru_checkpoint["hidden_dim_1"],
            hidden_dim_2=gru_checkpoint["hidden_dim_2"],
            dropout=gru_checkpoint["dropout"]
        ).to(device)
        gru_model.load_state_dict(gru_checkpoint["model_state"])
        gru_model.eval()

        with torch.no_grad():
            tx_tensor = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
            y_pred_gru_s = gru_model(tx_tensor).cpu().numpy()

        y_pred_gru = self.prep.target_scaler.inverse_transform(y_pred_gru_s.reshape(-1, 1)).flatten()
        results["GRU"] = self._compute_metrics(y_test_orig, y_pred_gru)

        # 6. Save JSON Metrics
        metrics_json_path = os.path.join(self.metrics_dir, "model_benchmark_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Evaluation metrics saved to: {metrics_json_path}")

        # 7. Generate Evaluation Figures
        self._plot_actual_vs_predicted(y_test_orig, {
            "Naive": y_pred_naive,
            "Random Forest": y_pred_rf,
            "LSTM": y_pred_lstm,
            "GRU": y_pred_gru
        })

        self._plot_loss_curves(
            lstm_res["train_losses"], lstm_res["val_losses"],
            gru_res["train_losses"], gru_res["val_losses"]
        )

        return results

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = float(calculate_mape(y_true, y_pred))
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(r2, 4),
            "MAPE_Percent": round(mape, 2)
        }

    def _plot_actual_vs_predicted(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="Actual Congestion Score", color="black", linewidth=2.5, linestyle="--")
        
        colors = {"Naive": "gray", "Random Forest": "blue", "LSTM": "red", "GRU": "green"}
        for model_name, y_pred in predictions.items():
            plt.plot(y_pred, label=f"Predicted ({model_name})", color=colors.get(model_name, "purple"), alpha=0.85, linewidth=1.8)

        plt.title("Test Set: Actual vs. Predicted Traffic Congestion Score", fontsize=14, fontweight="bold")
        plt.xlabel("Time Step (Test Index)", fontsize=12)
        plt.ylabel("Congestion Score [0.0 - 1.0]", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        
        save_path = os.path.join(self.fig_dir, "actual_vs_predicted.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Actual vs Predicted plot saved to: {save_path}")

    def _plot_loss_curves(self, lstm_tr: List[float], lstm_val: List[float], gru_tr: List[float], gru_val: List[float]):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(lstm_tr, label="Train Loss", color="navy")
        axes[0].plot(lstm_val, label="Val Loss", color="firebrick")
        axes[0].set_title("LSTM Training & Validation Loss", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        axes[1].plot(gru_tr, label="Train Loss", color="navy")
        axes[1].plot(gru_val, label="Val Loss", color="forestgreen")
        axes[1].set_title("GRU Training & Validation Loss", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE Loss")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        save_path = os.path.join(self.fig_dir, "loss_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Loss curves saved to: {save_path}")

if __name__ == "__main__":
    evaluator = Evaluator()
    res = evaluator.run_full_evaluation()
    print("\n=== MODEL BENCHMARK RESULTS ===")
    print(json.dumps(res, indent=4))
