"""
Custom LSTM neural network architecture for stock return direction prediction.

Architecture design:
  - Stacked LSTM layers capture temporal dependencies at multiple timescales
  - Batch normalization stabilizes training and reduces internal covariate shift
  - Dropout between layers reduces overfitting on financial data (high noise)
  - Two-layer MLP head maps LSTM output to class probabilities

AI-generated with Claude Code; reviewed and adapted by student.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMPredictor(nn.Module):
    """
    Custom stacked LSTM with dropout and batch normalization.
    Substantially designed by student; not a pretrained model.

    Input:  (batch, seq_len, input_size)
    Output: (batch, num_classes) logits
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # Stacked LSTM - dropout is applied between LSTM layers (not on last layer output)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.num_directions
        self.batch_norm = nn.BatchNorm1d(lstm_output_size)
        self.dropout = nn.Dropout(dropout)

        # Two-layer MLP head
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(lstm_output_size // 2, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for linear layers; orthogonal for LSTM weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 to encourage memory retention at start
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        lstm_out, _ = self.lstm(x, hidden)
        # Use only last timestep's output for prediction
        last_out = lstm_out[:, -1, :]
        out = self.batch_norm(last_out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities (useful for ensemble weighting)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        n_params = self.count_parameters()
        return (
            f"LSTMPredictor("
            f"hidden={self.hidden_size}, "
            f"layers={self.num_layers}, "
            f"params={n_params:,})"
        )
