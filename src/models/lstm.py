"""HeadwayLSTM — flat LSTM encoder for multi-bus headway forecasting.

Architecture decisions (from design):
  AD-1: Input is a flat concatenation of headway vector (max_N) and context (5)
        at each timestep. The caller performs the concatenation before calling
        forward. input_size = max_N + context_size.
  AD-2: Only the last hidden state h[-1] is used → Linear head → (B, output_size).
        seq2one: T_out=1 throughout Fase 5.
  AD-3: Mask is applied ONLY at the loss level. The model always produces
        output for all positions regardless of which slots are valid.

Invariants:
  INV-NO-COMPILE: torch.compile() is NOT used (Kaggle CUDA compatibility).
  INV-MASK: masked_mse_loss gates gradient on mask; model is mask-agnostic.

ACs covered: AC-MODEL-1..5, AC-LOSS-1..5.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HeadwayLSTM(nn.Module):
    """Flat LSTM encoder for multi-bus headway forecasting.

    Parameters
    ----------
    input_size:
        Size of the LSTM input at each timestep.
        Must equal max_N + context_size at the call site (AD-1, INV-INPUT-SIZE).
    hidden_size:
        Number of LSTM hidden units.
    output_size:
        Number of output positions (= max_N). The linear head maps
        hidden_size → output_size.
    num_layers:
        Number of stacked LSTM layers (default 1).
    dropout:
        Dropout probability applied between LSTM layers (default 0.0).
        Ignored when num_layers == 1 (PyTorch behaviour).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            (B, T_in, input_size) — concatenated headway + context tensor.
            Caller is responsible for torch.cat([input, context], dim=-1).

        Returns
        -------
        (B, output_size) — predicted next headway vector (z-scored).
        Uses only the last hidden state (AD-2).
        """
        # lstm_out: (B, T_in, hidden_size)
        # h_n:      (num_layers, B, hidden_size)
        _lstm_out, (h_n, _c_n) = self.lstm(x)

        # Take the hidden state from the topmost layer at the last timestep.
        # h_n[-1]: (B, hidden_size)
        last_hidden = h_n[-1]

        # Project to output space: (B, output_size)
        return self.head(last_hidden)


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error computed only over valid (mask == True) positions.

    Parameters
    ----------
    pred:
        (B, max_N) float32 — model predictions (z-scored).
    target:
        (B, max_N) float32 — ground-truth values (z-scored).
    mask:
        (B, max_N) bool — True = VALID position; False = absent/padded slot.

    Returns
    -------
    Scalar 0-dim tensor. If no position is True, returns 0.0 (clamp(min=1)
    prevents zero-division, per AD-3 and INV-MASK).

    Formula: ((pred - target)^2 * mask.float()).sum() / mask.float().sum().clamp(min=1)
    """
    mask_f = mask.float()
    squared_error = (pred - target) ** 2
    # Sum only over valid positions, then normalize by count (clamped to 1).
    return (squared_error * mask_f).sum() / mask_f.sum().clamp(min=1)
