"""
Step 6.3 — LSTM baseline model.

Two stacked LSTM layers (hidden_size → hidden_size/2) followed by four
independent linear heads, one per forecast horizon (3hr, 12hr, 24hr, 72hr).
Output is a point forecast for PM2.5 at each horizon.

Input shape:  (batch, seq_len, n_features)
Output shape: (batch, 4)   — columns ordered [h3, h12, h24, h72]
"""
from __future__ import annotations

import torch
import torch.nn as nn

HORIZONS = [3, 12, 24, 72]   # forecast horizons in hours


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Four heads share the same LSTM representation; each is independent.
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in HORIZONS
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]   # (batch, hidden_size) — final timestep
        return torch.cat([head(last) for head in self.heads], dim=1)
        # returns (batch, 4)
