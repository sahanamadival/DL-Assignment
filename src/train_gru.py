import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.config import load_config
from src.preprocessing import ChronologicalPreprocessor
from src.sequence_builder import SequenceBuilder

class GRUModel(nn.Module):
    """2-layer PyTorch GRU for short-term traffic congestion forecasting."""

    def __init__(self, input_dim: int, hidden_dim_1: int = 64, hidden_dim_2: int = 32, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim_1, batch_first=True, num_layers=1)
        self.dropout1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(input_size=hidden_dim_1, hidden_size=hidden_dim_2, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim_2, 16)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.gru1(x)
        out1 = self.dropout1(out1)
        out2, h_n = self.gru2(out1)
        # Take last time step output
        last_out = out2[:, -1, :]
        last_out = self.dropout2(last_out)
        dense1 = self.relu(self.fc1(last_out))
        prediction = self.fc_out(dense1)
        return prediction.squeeze(-1)

class GRUTrainer:
    """Trainer pipeline for GRU congestion model."""

    def __init__(self):
        self.cfg = load_config()
        self.prep = ChronologicalPreprocessor()
        self.builder = SequenceBuilder()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.detector.device == "cuda" else "cpu")

    def train_model(self) -> Dict[str, Any]:
        df = self.prep.load_raw_csv()
        train_df, val_df, test_df = self.prep.split_chronologically(df)

        X_train_s, y_train_s = self.prep.fit_transform_train(train_df)
        X_val_s, y_val_s = self.prep.transform_eval(val_df)
        X_test_s, y_test_s = self.prep.transform_eval(test_df)

        # Build 3D sequences
        X_tr_seq, y_tr_seq = self.builder.create_sequences(X_train_s, y_train_s)
        X_val_seq, y_val_seq = self.builder.create_sequences(X_val_s, y_val_s)
        X_te_seq, y_te_seq = self.builder.create_sequences(X_test_s, y_test_s)

        # PyTorch DataLoaders
        train_ds = TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32), torch.tensor(y_tr_seq, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=self.cfg.model_training.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.model_training.batch_size, shuffle=False)

        input_dim = X_tr_seq.shape[2]
        model = GRUModel(
            input_dim=input_dim,
            hidden_dim_1=self.cfg.model_training.hidden_dim_1,
            hidden_dim_2=self.cfg.model_training.hidden_dim_2,
            dropout=self.cfg.model_training.dropout
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.model_training.learning_rate)

        epochs = self.cfg.model_training.epochs
        patience = self.cfg.model_training.patience
        best_val_loss = float('inf')
        patience_counter = 0

        train_losses, val_losses = [], []
        best_model_state = None

        print(f"[INFO] Training GRU model for {epochs} epochs on {self.device}...")
        for epoch in range(epochs):
            model.train()
            tr_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * len(by)
            tr_loss /= len(train_ds)
            train_losses.append(tr_loss)

            # Validation step
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    out = model(bx)
                    loss = criterion(out, by)
                    v_loss += loss.item() * len(by)
            v_loss /= len(val_ds)
            val_losses.append(v_loss)

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Train Loss: {tr_loss:.6f} | Val Loss: {v_loss:.6f}")

            # Early stopping check
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save trained GRU model
        out_dir = self.cfg.paths.gru_model_dir
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "gru_congestion.pt")
        torch.save({
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim_1": self.cfg.model_training.hidden_dim_1,
            "hidden_dim_2": self.cfg.model_training.hidden_dim_2,
            "dropout": self.cfg.model_training.dropout,
            "train_losses": train_losses,
            "val_losses": val_losses
        }, model_path)
        print(f"[INFO] GRU model checkpoint saved to: {model_path}")

        return {
            "model_path": model_path,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_data": (X_te_seq, y_te_seq)
        }

if __name__ == "__main__":
    trainer = GRUTrainer()
    res = trainer.train_model()
    print("GRU Training Completed! Best Val Loss:", res["best_val_loss"])
