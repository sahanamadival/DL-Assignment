import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set reproducible seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class PyTorchLSTM(nn.Module):
    """2-layer PyTorch LSTM for METR-LA traffic speed forecasting."""

    def __init__(self, input_dim: int = 1, hidden_dim_1: int = 64, hidden_dim_2: int = 32, dropout: float = 0.2):
        super(PyTorchLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim_1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_dim_1, hidden_size=hidden_dim_2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim_2, 16)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        out1 = self.drop1(out1)
        out2, _ = self.lstm2(out1)
        last_step = out2[:, -1, :]
        last_step = self.drop2(last_step)
        dense1 = self.relu(self.fc1(last_step))
        out = self.fc_out(dense1)
        return out.squeeze(-1)

class PyTorchGRU(nn.Module):
    """2-layer PyTorch GRU for METR-LA traffic speed forecasting."""

    def __init__(self, input_dim: int = 1, hidden_dim_1: int = 64, hidden_dim_2: int = 32, dropout: float = 0.2):
        super(PyTorchGRU, self).__init__()
        self.gru1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim_1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(input_size=hidden_dim_1, hidden_size=hidden_dim_2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim_2, 16)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.gru1(x)
        out1 = self.drop1(out1)
        out2, _ = self.gru2(out1)
        last_step = out2[:, -1, :]
        last_step = self.drop2(last_step)
        dense1 = self.relu(self.fc1(last_step))
        out = self.fc_out(dense1)
        return out.squeeze(-1)

class DeepModelTrainer:
    """Trainer pipeline for PyTorch LSTM and GRU models across 15m, 30m, 45m horizons."""

    def __init__(self, seq_dir: str = "data/processed/sequences"):
        self.seq_dir = seq_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model_dir = "models/lstm"
        self.gru_model_dir = "models/gru"
        self.fig_dir = "results/figures"

        os.makedirs(self.lstm_model_dir, exist_ok=True)
        os.makedirs(self.gru_model_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def train_all_deep_models(self) -> Dict[str, Any]:
        horizons = {
            "15min": {"file": "seq_15min.npz", "label": "15min"},
            "30min": {"file": "seq_30min.npz", "label": "30min"},
            "45min": {"file": "seq_45min.npz", "label": "45min"}
        }

        model_types = ["lstm", "gru"]
        summary_report = {
            "device": str(self.device),
            "architecture": {
                "LSTM": "Input(12,1) -> LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, ReLU) -> Dense(1)",
                "GRU": "Input(12,1) -> GRU(64) -> Dropout(0.2) -> GRU(32) -> Dropout(0.2) -> Dense(16, ReLU) -> Dense(1)"
            },
            "models_trained": {}
        }

        print(f"=== STAGE 6 — TRAINING PYTORCH LSTM & GRU MODELS ON {self.device} ===")

        for m_type in model_types:
            for h_key, h_info in horizons.items():
                npz_path = os.path.join(self.seq_dir, h_info["file"])
                if not os.path.exists(npz_path):
                    raise FileNotFoundError(f"Sequence npz file missing at: {npz_path}")

                data = np.load(npz_path)
                X_tr, y_tr = data["X_train"], data["y_train"]
                X_v, y_v = data["X_val"], data["y_val"]

                model_key = f"{m_type}_{h_key}"
                print(f"\n[INFO] Training {m_type.upper()} ({h_key}) -> Train samples: {len(X_tr)}, Val samples: {len(X_v)}")

                res = self._train_single_model(
                    model_type=m_type,
                    horizon_key=h_key,
                    X_tr=X_tr, y_tr=y_tr,
                    X_v=X_v, y_v=y_v
                )
                summary_report["models_trained"][model_key] = res

        return summary_report

    def _train_single_model(
        self,
        model_type: str,
        horizon_key: str,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_v: np.ndarray, y_v: np.ndarray,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 7,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        
        # PyTorch DataLoaders
        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr.flatten(), dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_v, dtype=torch.float32), torch.tensor(y_v.flatten(), dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        input_dim = X_tr.shape[2]
        if model_type == "lstm":
            model = PyTorchLSTM(input_dim=input_dim).to(self.device)
            save_dir = self.lstm_model_dir
        else:
            model = PyTorchGRU(input_dim=input_dim).to(self.device)
            save_dir = self.gru_model_dir

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state = None

        train_losses = []
        val_losses = []

        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            model.train()
            tr_loss_sum = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                tr_loss_sum += loss.item() * len(by)

            ep_tr_loss = tr_loss_sum / len(train_ds)
            train_losses.append(ep_tr_loss)

            # Validation
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    out = model(bx)
                    loss = criterion(out, by)
                    val_loss_sum += loss.item() * len(by)

            ep_val_loss = val_loss_sum / len(val_ds)
            val_losses.append(ep_val_loss)

            if epoch % 5 == 0 or epoch == 1 or epoch == max_epochs:
                print(f"  Epoch [{epoch:02d}/{max_epochs}] - Train Loss: {ep_tr_loss:.6f} | Val Loss: {ep_val_loss:.6f}")

            # Early stopping check
            if ep_val_loss < best_val_loss:
                best_val_loss = ep_val_loss
                best_epoch = epoch
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [INFO] Early stopping triggered at epoch {epoch} (Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f})")
                    break

        elapsed_sec = round(time.time() - start_time, 2)

        # Save model checkpoint
        save_path = os.path.join(save_dir, f"{model_type}_{horizon_key}.pt")
        if best_state is not None:
            model.load_state_dict(best_state)

        torch.save({
            "model_type": model_type,
            "horizon_key": horizon_key,
            "model_state": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "training_time_sec": elapsed_sec
        }, save_path)
        print(f"  [INFO] Checkpoint saved: {save_path}")

        # Plot training curve
        fig_path = os.path.join(self.fig_dir, f"{model_type}_{horizon_key}_training.png")
        self._plot_training_curve(model_type, horizon_key, train_losses, val_losses, fig_path)

        return {
            "model_type": model_type.upper(),
            "horizon": horizon_key,
            "train_samples": len(X_tr),
            "val_samples": len(X_v),
            "epochs_completed": len(train_losses),
            "best_epoch": best_epoch,
            "best_val_loss": round(float(best_val_loss), 6),
            "early_stopping_triggered": (patience_counter >= patience),
            "training_time_sec": elapsed_sec,
            "saved_model_path": save_path,
            "training_curve_path": fig_path
        }

    def _plot_training_curve(self, model_type: str, horizon_key: str, train_losses: List[float], val_losses: List[float], out_path: str):
        plt.figure(figsize=(8, 4.5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss (MSE)", color="#2563eb", linewidth=2.0)
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss (MSE)", color="#dc2626", linewidth=2.0, linestyle="--")

        plt.title(f"{model_type.upper()} ({horizon_key} Horizon) — Training & Validation Loss Curve", fontsize=11, fontweight="bold")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("MSE Loss (Scaled Target)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    trainer = DeepModelTrainer()
    report = trainer.train_all_deep_models()
    print("\n=== STAGE 6 TRAINING REPORT SUMMARY ===")
    for m_key, m_val in report["models_trained"].items():
        print(f"\nModel: {m_key}")
        for k, v in m_val.items():
            print(f"  {k}: {v}")
