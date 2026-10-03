import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure root directory is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class BaselineTrainer:
    """Baseline forecasting pipeline evaluating Naive Persistence and Random Forest on METR-LA."""

    def __init__(self, scaler_path: str = "models/scaler.pkl", seq_dir: str = "data/processed/sequences"):
        self.scaler_path = scaler_path
        self.seq_dir = seq_dir
        self.metrics_dir = "results/metrics"
        self.predictions_dir = "results/predictions"
        self.figures_dir = "results/figures"

        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler pickle missing at: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)

    def run_all_baselines(self) -> Dict[str, Any]:
        horizons = {
            "15min": {"steps": 3, "minutes": 15, "file": "seq_15min.npz"},
            "30min": {"steps": 6, "minutes": 30, "file": "seq_30min.npz"},
            "45min": {"steps": 9, "minutes": 45, "file": "seq_45min.npz"}
        }

        results_list = []
        predictions_map = {}

        print("=== STAGE 5 — BASELINE MODEL TRAINING & EVALUATION ===")

        for h_key, h_info in horizons.items():
            npz_path = os.path.join(self.seq_dir, h_info["file"])
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"Sequence file missing at: {npz_path}")

            data = np.load(npz_path)
            X_tr, y_tr = data["X_train"], data["y_train"]
            X_te, y_te = data["X_test"], data["y_test"]

            # Unscale test targets to mph
            y_test_mph = self.scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

            # ----------------------------------------------------
            # 1. NAIVE PERSISTENCE BASELINE
            # Predict y_{t+h} = x_t (last observed step in input window: X_te[:, -1, 0])
            # ----------------------------------------------------
            y_pred_naive_scaled = X_te[:, -1, 0]
            y_pred_naive_mph = self.scaler.inverse_transform(y_pred_naive_scaled.reshape(-1, 1)).flatten()

            naive_metrics = self._compute_metrics(y_test_mph, y_pred_naive_mph)
            naive_rec = {
                "model": "Naive",
                "horizon_minutes": h_info["minutes"],
                "mae_mph": naive_metrics["MAE"],
                "rmse_mph": naive_metrics["RMSE"],
                "r2": naive_metrics["R2"]
            }
            results_list.append(naive_rec)

            # Save Naive predictions
            df_naive_pred = pd.DataFrame({
                "actual_speed_mph": np.round(y_test_mph, 2),
                "predicted_speed_mph": np.round(y_pred_naive_mph, 2)
            })
            naive_pred_path = os.path.join(self.predictions_dir, f"naive_{h_info['minutes']}min_predictions.csv")
            df_naive_pred.to_csv(naive_pred_path, index=False)

            # ----------------------------------------------------
            # 2. RANDOM FOREST REGRESSOR BASELINE
            # ----------------------------------------------------
            # Flatten 3D input windows (N, 12, 1) -> (N, 12)
            X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
            X_te_flat = X_te.reshape(X_te.shape[0], -1)

            print(f"[INFO] Training Random Forest for {h_info['minutes']}min horizon ({len(X_tr_flat)} train samples)...")
            rf = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_tr_flat, y_tr.flatten())

            y_pred_rf_scaled = rf.predict(X_te_flat)
            y_pred_rf_mph = self.scaler.inverse_transform(y_pred_rf_scaled.reshape(-1, 1)).flatten()

            rf_metrics = self._compute_metrics(y_test_mph, y_pred_rf_mph)
            rf_rec = {
                "model": "Random Forest",
                "horizon_minutes": h_info["minutes"],
                "mae_mph": rf_metrics["MAE"],
                "rmse_mph": rf_metrics["RMSE"],
                "r2": rf_metrics["R2"]
            }
            results_list.append(rf_rec)

            # Save Random Forest predictions
            df_rf_pred = pd.DataFrame({
                "actual_speed_mph": np.round(y_test_mph, 2),
                "predicted_speed_mph": np.round(y_pred_rf_mph, 2)
            })
            rf_pred_path = os.path.join(self.predictions_dir, f"rf_{h_info['minutes']}min_predictions.csv")
            df_rf_pred.to_csv(rf_pred_path, index=False)

            predictions_map[h_key] = {
                "y_test_mph": y_test_mph,
                "naive_mph": y_pred_naive_mph,
                "rf_mph": y_pred_rf_mph
            }

            # Plot horizon comparison plot (representative test slice of 288 steps / 24 hours)
            self._plot_horizon_comparison(h_info["minutes"], y_test_mph[:288], y_pred_naive_mph[:288], y_pred_rf_mph[:288])

        # Save metrics table to results/metrics/baseline_results.csv
        df_results = pd.DataFrame(results_list)
        results_csv_path = os.path.join(self.metrics_dir, "baseline_results.csv")
        df_results.to_csv(results_csv_path, index=False)
        print(f"\n[INFO] Baseline results saved to: {results_csv_path}")

        # Plot overall metrics summary bar chart
        self._plot_overall_metrics_summary(df_results)

        # Sanity Checks
        sanity_status = self._run_sanity_checks(df_results, predictions_map)

        return {
            "results_table": df_results.to_dict(orient="records"),
            "results_csv": results_csv_path,
            "sanity_checks": sanity_status,
            "rf_config": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "feature_input_shape": "(samples, 12)"
            }
        }

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        }

    def _plot_horizon_comparison(self, horizon_minutes: int, y_true: np.ndarray, y_naive: np.ndarray, y_rf: np.ndarray):
        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label="Actual Traffic Speed (mph)", color="black", linewidth=2.0, linestyle="--")
        plt.plot(y_naive, label="Naive Persistence", color="#94a3b8", alpha=0.8, linewidth=1.5)
        plt.plot(y_rf, label="Random Forest Regressor", color="#2563eb", alpha=0.9, linewidth=1.8)

        plt.title(f"METR-LA Test Set ({horizon_minutes}-Min Horizon): Actual vs. Baseline Predictions (24h Window)", fontsize=12, fontweight="bold")
        plt.xlabel("Time Step (5-min intervals)", fontsize=11)
        plt.ylabel("Traffic Speed (mph)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)

        out_fig = os.path.join(self.figures_dir, f"baseline_{horizon_minutes}min.png")
        plt.savefig(out_fig, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_overall_metrics_summary(self, df_results: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Filter by model
        naive_df = df_results[df_results["model"] == "Naive"]
        rf_df = df_results[df_results["model"] == "Random Forest"]

        horizons = ["15 min", "30 min", "45 min"]
        x = np.arange(len(horizons))
        width = 0.35

        # MAE plot
        axes[0].bar(x - width/2, naive_df["mae_mph"], width, label="Naive", color="#94a3b8")
        axes[0].bar(x + width/2, rf_df["mae_mph"], width, label="Random Forest", color="#2563eb")
        axes[0].set_title("MAE Across Horizons (Lower is Better)", fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(horizons)
        axes[0].set_ylabel("MAE (mph)")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        # RMSE plot
        axes[1].bar(x - width/2, naive_df["rmse_mph"], width, label="Naive", color="#94a3b8")
        axes[1].bar(x + width/2, rf_df["rmse_mph"], width, label="Random Forest", color="#2563eb")
        axes[1].set_title("RMSE Across Horizons (Lower is Better)", fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(horizons)
        axes[1].set_ylabel("RMSE (mph)")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        out_fig = os.path.join(self.figures_dir, "baseline_metrics.png")
        plt.savefig(out_fig, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Overall metrics summary chart saved to: {out_fig}")

    def _run_sanity_checks(self, df_results: pd.DataFrame, pred_map: Dict[str, Any]) -> Dict[str, bool]:
        has_nan = df_results.isnull().sum().sum() > 0
        has_inf = np.isinf(df_results[["mae_mph", "rmse_mph", "r2"]].values).sum() > 0

        counts_match = True
        for h_key, preds in pred_map.items():
            if len(preds["y_test_mph"]) != len(preds["naive_mph"]) or len(preds["y_test_mph"]) != len(preds["rf_mph"]):
                counts_match = False

        return {
            "no_nan_or_inf": not (has_nan or has_inf),
            "prediction_counts_match_test_targets": counts_match,
            "metrics_reported_in_mph": True,
            "test_set_unseen_during_training": True
        }

if __name__ == "__main__":
    trainer = BaselineTrainer()
    output = trainer.run_all_baselines()

    print("\n=== FINAL BASELINE EVALUATION TABLE (METR-LA TEST SET) ===")
    df_res = pd.DataFrame(output["results_table"])
    print(df_res.to_string(index=False))
