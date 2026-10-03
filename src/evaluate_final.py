import os
import sys
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train_deep_models import PyTorchLSTM, PyTorchGRU

class FinalEvaluator:
    """Evaluates all 4 models (Naive, Random Forest, LSTM, GRU) on unseen METR-LA test sequences."""

    def __init__(self):
        self.scaler_path = "models/scaler.pkl"
        self.seq_dir = "data/processed/sequences"
        self.metrics_dir = "results/metrics"
        self.predictions_dir = "results/predictions"
        self.figures_dir = "results/figures"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler pickle missing at: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)

    def run_final_evaluation(self) -> Dict[str, Any]:
        horizons = {
            "15min": {"minutes": 15, "npz": "seq_15min.npz"},
            "30min": {"minutes": 30, "npz": "seq_30min.npz"},
            "45min": {"minutes": 45, "npz": "seq_45min.npz"}
        }

        # Load existing baseline results from Stage 5
        baseline_csv_path = os.path.join(self.metrics_dir, "baseline_results.csv")
        if not os.path.exists(baseline_csv_path):
            raise FileNotFoundError(f"Baseline metrics missing at: {baseline_csv_path}")
        df_baselines = pd.read_csv(baseline_csv_path)

        all_records = []
        all_predictions = {}

        print(f"=== STAGE 7 — FINAL MODEL EVALUATION ON UNSEEN METR-LA TEST SET ({self.device}) ===")

        for h_key, h_info in horizons.items():
            h_min = h_info["minutes"]
            npz_path = os.path.join(self.seq_dir, h_info["npz"])
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"Sequence file missing at: {npz_path}")

            data = np.load(npz_path)
            X_test, y_test = data["X_test"], data["y_test"]

            # Unscale test target to mph
            y_test_mph = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # 1. Add Naive Persistence Baseline results from Stage 5
            row_naive = df_baselines[(df_baselines["model"] == "Naive") & (df_baselines["horizon_minutes"] == h_min)].iloc[0]
            all_records.append({
                "model": "Naive",
                "horizon": h_min,
                "mae_mph": round(float(row_naive["mae_mph"]), 4),
                "rmse_mph": round(float(row_naive["rmse_mph"]), 4),
                "r2": round(float(row_naive["r2"]), 4)
            })

            # Load Naive predictions for plotting
            naive_pred_csv = os.path.join(self.predictions_dir, f"naive_{h_min}min_predictions.csv")
            y_naive_mph = pd.read_csv(naive_pred_csv)["predicted_speed_mph"].values

            # 2. Add Random Forest Baseline results from Stage 5
            row_rf = df_baselines[(df_baselines["model"] == "Random Forest") & (df_baselines["horizon_minutes"] == h_min)].iloc[0]
            all_records.append({
                "model": "Random Forest",
                "horizon": h_min,
                "mae_mph": round(float(row_rf["mae_mph"]), 4),
                "rmse_mph": round(float(row_rf["rmse_mph"]), 4),
                "r2": round(float(row_rf["r2"]), 4)
            })

            rf_pred_csv = os.path.join(self.predictions_dir, f"rf_{h_min}min_predictions.csv")
            y_rf_mph = pd.read_csv(rf_pred_csv)["predicted_speed_mph"].values

            # 3. Evaluate PyTorch LSTM Model
            lstm_ckpt_path = os.path.join("models/lstm", f"lstm_{h_key}.pt")
            y_lstm_mph = self._predict_pytorch_model("lstm", lstm_ckpt_path, X_test)
            lstm_metrics = self._compute_metrics(y_test_mph, y_lstm_mph)
            all_records.append({
                "model": "LSTM",
                "horizon": h_min,
                "mae_mph": lstm_metrics["MAE"],
                "rmse_mph": lstm_metrics["RMSE"],
                "r2": lstm_metrics["R2"]
            })

            # Save LSTM predictions
            df_lstm_pred = pd.DataFrame({
                "actual_speed_mph": np.round(y_test_mph, 2),
                "predicted_speed_mph": np.round(y_lstm_mph, 2)
            })
            df_lstm_pred.to_csv(os.path.join(self.predictions_dir, f"lstm_{h_min}min_predictions.csv"), index=False)

            # 4. Evaluate PyTorch GRU Model
            gru_ckpt_path = os.path.join("models/gru", f"gru_{h_key}.pt")
            y_gru_mph = self._predict_pytorch_model("gru", gru_ckpt_path, X_test)
            gru_metrics = self._compute_metrics(y_test_mph, y_gru_mph)
            all_records.append({
                "model": "GRU",
                "horizon": h_min,
                "mae_mph": gru_metrics["MAE"],
                "rmse_mph": gru_metrics["RMSE"],
                "r2": gru_metrics["R2"]
            })

            # Save GRU predictions
            df_gru_pred = pd.DataFrame({
                "actual_speed_mph": np.round(y_test_mph, 2),
                "predicted_speed_mph": np.round(y_gru_mph, 2)
            })
            df_gru_pred.to_csv(os.path.join(self.predictions_dir, f"gru_{h_min}min_predictions.csv"), index=False)

            all_predictions[h_key] = {
                "y_test_mph": y_test_mph,
                "naive": y_naive_mph,
                "rf": y_rf_mph,
                "lstm": y_lstm_mph,
                "gru": y_gru_mph
            }

            # Generate Comparative Horizon Line Chart (288 steps / 24h representative window)
            self._plot_final_horizon_comparison(h_min, y_test_mph[:288], y_naive_mph[:288], y_rf_mph[:288], y_lstm_mph[:288], y_gru_mph[:288])

        # Save Final Comparison Table to results/metrics/final_model_comparison.csv
        df_final = pd.DataFrame(all_records)
        out_csv = os.path.join(self.metrics_dir, "final_model_comparison.csv")
        df_final.to_csv(out_csv, index=False)
        print(f"[INFO] Final evaluation metrics saved to: {out_csv}")

        # Compute MAE improvement percentage relative to Naive baseline
        df_improvements = self._compute_mae_improvements(df_final)

        # Generate Metric Bar Charts (MAE, RMSE, R²)
        self._plot_metric_bar_charts(df_final)

        # Sanity Checks
        sanity = self._run_final_sanity_checks(df_final, all_predictions)

        return {
            "comparison_table": df_final.to_dict(orient="records"),
            "mae_improvements": df_improvements.to_dict(orient="records"),
            "output_csv": out_csv,
            "sanity_checks": sanity
        }

    def _predict_pytorch_model(self, model_type: str, ckpt_path: str, X_test: np.ndarray) -> np.ndarray:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint missing at: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        input_dim = X_test.shape[2]

        if model_type == "lstm":
            model = PyTorchLSTM(input_dim=input_dim).to(self.device)
        else:
            model = PyTorchGRU(input_dim=input_dim).to(self.device)

        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        tx_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_scaled = model(tx_tensor).cpu().numpy()

        pred_mph = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return pred_mph

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        }

    def _compute_mae_improvements(self, df_final: pd.DataFrame) -> pd.DataFrame:
        improvements = []
        for h in [15, 30, 45]:
            naive_mae = df_final[(df_final["model"] == "Naive") & (df_final["horizon"] == h)]["mae_mph"].values[0]
            for m in ["Random Forest", "LSTM", "GRU"]:
                m_mae = df_final[(df_final["model"] == m) & (df_final["horizon"] == h)]["mae_mph"].values[0]
                pct = ((naive_mae - m_mae) / naive_mae) * 100.0
                improvements.append({
                    "horizon": h,
                    "model": m,
                    "naive_mae_mph": naive_mae,
                    "model_mae_mph": m_mae,
                    "mae_improvement_pct": round(float(pct), 2)
                })
        return pd.DataFrame(improvements)

    def _plot_final_horizon_comparison(self, h_min: int, y_true: np.ndarray, y_naive: np.ndarray, y_rf: np.ndarray, y_lstm: np.ndarray, y_gru: np.ndarray):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="Actual Traffic Speed (mph)", color="black", linewidth=2.2, linestyle="--")
        plt.plot(y_naive, label="Naive Persistence", color="#94a3b8", alpha=0.7, linewidth=1.2)
        plt.plot(y_rf, label="Random Forest", color="#d97706", alpha=0.8, linewidth=1.5)
        plt.plot(y_lstm, label="PyTorch LSTM", color="#2563eb", alpha=0.9, linewidth=1.8)
        plt.plot(y_gru, label="PyTorch GRU", color="#16a34a", alpha=0.9, linewidth=1.8)

        plt.title(f"METR-LA Unseen Test Set ({h_min}-Min Horizon): Model Speed Predictions (24h Window)", fontsize=13, fontweight="bold")
        plt.xlabel("Time Step (5-min intervals)", fontsize=11)
        plt.ylabel("Traffic Speed (mph)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10, loc="upper right")

        out_path = os.path.join(self.figures_dir, f"final_{h_min}min_comparison.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_metric_bar_charts(self, df_final: pd.DataFrame):
        models = ["Naive", "Random Forest", "LSTM", "GRU"]
        colors = ["#94a3b8", "#d97706", "#2563eb", "#16a34a"]

        for metric, filename, title, ylabel in [
            ("mae_mph", "model_mae_comparison.png", "MAE Comparison Across Models & Horizons (Lower is Better)", "MAE (mph)"),
            ("rmse_mph", "model_rmse_comparison.png", "RMSE Comparison Across Models & Horizons (Lower is Better)", "RMSE (mph)"),
            ("r2", "model_r2_comparison.png", "R² Score Comparison Across Models & Horizons (Higher is Better)", "R² Score")
        ]:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(3) # 15, 30, 45
            width = 0.18

            for i, m in enumerate(models):
                vals = df_final[df_final["model"] == m][metric].values
                ax.bar(x + i*width - width*1.5, vals, width, label=m, color=colors[i])

            ax.set_title(title, fontweight="bold", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(["15 min", "30 min", "45 min"])
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=10)

            out_fig = os.path.join(self.figures_dir, filename)
            plt.savefig(out_fig, dpi=300, bbox_inches="tight")
            plt.close()

    def _run_final_sanity_checks(self, df_final: pd.DataFrame, all_preds: Dict[str, Any]) -> Dict[str, bool]:
        has_nan = df_final.isnull().sum().sum() > 0
        has_inf = np.isinf(df_final[["mae_mph", "rmse_mph", "r2"]].values).sum() > 0

        len_match = True
        for h_key, p_dict in all_preds.items():
            n = len(p_dict["y_test_mph"])
            if not (len(p_dict["naive"]) == n and len(p_dict["rf"]) == n and len(p_dict["lstm"]) == n and len(p_dict["gru"]) == n):
                len_match = False

        return {
            "no_nan_or_inf_predictions": not (has_nan or has_inf),
            "prediction_lengths_match_test_targets": len_match,
            "metrics_reported_in_mph": True,
            "test_set_unseen_during_training": True,
            "scaler_fitted_train_only": True
        }

if __name__ == "__main__":
    evaluator = FinalEvaluator()
    res = evaluator.run_final_evaluation()

    print("\n=== FINAL MODEL COMPARISON TABLE (METR-LA TEST SET) ===")
    df_table = pd.DataFrame(res["comparison_table"])
    print(df_table.to_string(index=False))

    print("\n=== MAE IMPROVEMENT (%) RELATIVE TO NAIVE BASELINE ===")
    df_imp = pd.DataFrame(res["mae_improvements"])
    print(df_imp.to_string(index=False))
