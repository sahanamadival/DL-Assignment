import numpy as np
from typing import Tuple, Optional
from src.config import load_config, TimeSeriesConfig

class SequenceBuilder:
    """Converts 2D scaled feature arrays into 3D sliding window tensors for temporal models."""

    def __init__(self, time_series_cfg: Optional[TimeSeriesConfig] = None):
        cfg = load_config()
        self.ts_cfg = time_series_cfg or cfg.time_series
        self.seq_len = self.ts_cfg.sequence_length
        self.horizon = self.ts_cfg.prediction_horizon

    def create_sequences(self, X_scaled: np.ndarray, y_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generates 3D input sequences X of shape (num_samples, sequence_length, num_features) and target array y of shape (num_samples,).

        For index i:
            X[i] = X_scaled[i : i + sequence_length]
            y[i] = y_scaled[i + sequence_length + horizon - 1] (or single step target if horizon=1)
        """
        X_seq, y_seq = [], []
        n_samples = len(X_scaled)
        
        # Ensure horizon offset stays within bounds
        horizon_offset = max(1, min(self.horizon, 15))

        for i in range(n_samples - self.seq_len - horizon_offset + 1):
            X_window = X_scaled[i : i + self.seq_len]
            target_val = y_scaled[i + self.seq_len + horizon_offset - 1]
            
            X_seq.append(X_window)
            y_seq.append(target_val)

        X_tensor = np.array(X_seq, dtype=np.float32)
        y_tensor = np.array(y_seq, dtype=np.float32)

        return X_tensor, y_tensor

if __name__ == "__main__":
    builder = SequenceBuilder()
    dummy_X = np.random.randn(100, 10).astype(np.float32)
    dummy_y = np.random.randn(100).astype(np.float32)
    
    X_seq, y_seq = builder.create_sequences(dummy_X, dummy_y)
    print("Sequence Builder test:")
    print("X sequence tensor shape:", X_seq.shape)
    print("y sequence tensor shape:", y_seq.shape)
