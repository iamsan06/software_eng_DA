"""PyTorch Sequential Deep Learning Architectures (LSTM and GRU) for Lag-Aware Crop Yield Forecasting."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CropYieldSequenceDataset(Dataset):
    """Dataset packaging 16-week temporal rainfall sequences and static spatial context."""

    def __init__(
        self,
        sequence_data: np.ndarray,
        static_data: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        sequence_data : np.ndarray
            Shape (N, 16, F_seq) where 16 is the pre-harvest week sequence.
        static_data : Optional[np.ndarray]
            Shape (N, F_static) containing spatial one-hot or seasonal scalars.
        targets : Optional[np.ndarray]
            Shape (N,) containing crop yield values.
        """
        self.seq_tensors = torch.tensor(sequence_data, dtype=torch.float32)
        if static_data is not None:
            self.static_tensors = torch.tensor(static_data, dtype=torch.float32)
        else:
            self.static_tensors = None

        if targets is not None:
            self.target_tensors = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        else:
            self.target_tensors = None

    def __len__(self) -> int:
        return len(self.seq_tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.static_tensors is not None and self.target_tensors is not None:
            return self.seq_tensors[idx], self.static_tensors[idx], self.target_tensors[idx]
        elif self.target_tensors is not None:
            return self.seq_tensors[idx], self.target_tensors[idx]
        return self.seq_tensors[idx]


class RiceLSTM(nn.Module):
    """Bidirectional/Unidirectional multi-layer LSTM for 16-week pre-harvest sequence regression."""

    def __init__(
        self,
        seq_input_dim: int = 1,
        static_input_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_in_dim = hidden_dim * self.num_directions + static_input_dim
        self.head = nn.Sequential(
            nn.Linear(fc_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, seq_x: torch.Tensor, static_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        seq_x: (B, 16, seq_input_dim)
        static_x: (B, static_input_dim)
        """
        out, (hn, cn) = self.lstm(seq_x)

        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = hn[-2, :, :]
            h_backward = hn[-1, :, :]
            h_last = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_last = hn[-1, :, :]

        if static_x is not None:
            combined = torch.cat([h_last, static_x], dim=1)
        else:
            combined = h_last

        pred = self.head(combined)
        return pred


class RiceGRU(nn.Module):
    """Gated Recurrent Unit (GRU) network for sequential pre-harvest rainfall modeling."""

    def __init__(
        self,
        seq_input_dim: int = 1,
        static_input_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        fc_in_dim = hidden_dim + static_input_dim
        self.head = nn.Sequential(
            nn.Linear(fc_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, seq_x: torch.Tensor, static_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, hn = self.gru(seq_x)
        h_last = hn[-1, :, :]

        if static_x is not None:
            combined = torch.cat([h_last, static_x], dim=1)
        else:
            combined = h_last

        pred = self.head(combined)
        return pred


class SequenceTrainer:
    """Trainer handling optimization, validation loss monitoring, and early stopping for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.patience = patience
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict[str, Any]] = None

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            seq_x, static_x, target = [t.to(self.device) for t in batch]
            self.optimizer.zero_grad()
            preds = self.model(seq_x, static_x)
            loss = self.criterion(preds, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(1, n_batches)

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Compute validation MSE loss and MAE."""
        self.model.eval()
        total_loss = 0.0
        total_abs_err = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in loader:
                seq_x, static_x, target = [t.to(self.device) for t in batch]
                preds = self.model(seq_x, static_x)
                loss = self.criterion(preds, target)
                total_loss += float(loss.item()) * len(target)
                total_abs_err += float(torch.sum(torch.abs(preds - target)).item())
                n_samples += len(target)

        val_mse = total_loss / max(1, n_samples)
        val_mae = total_abs_err / max(1, n_samples)
        return val_mse, val_mae

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 50,
    ) -> List[Dict[str, float]]:
        history = []
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_mse, val_mae = self.evaluate(val_loader)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mse": val_mse,
                "val_mae": val_mae,
            })

            if val_mae < self.best_val_loss:
                self.best_val_loss = val_mae
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return history

    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds_list = []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    seq_x, static_x, _ = [t.to(self.device) for t in batch]
                elif len(batch) == 2:
                    seq_x, static_x = [t.to(self.device) for t in batch]
                else:
                    seq_x = batch[0].to(self.device)
                    static_x = None
                preds = self.model(seq_x, static_x)
                preds_list.append(preds.cpu().numpy())
        return np.vstack(preds_list).flatten()
