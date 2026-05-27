"""SpatialConvLSTM — 1D spatial conv + LSTM temporal encoder for headway forecasting.

Architecture decisions applied (from design doc):
  AD-1: Forward signature is (inp, ctx, input_mask) — 3 separate tensors.
        inp: (B, T_in, max_N), ctx: (B, T_in, context_size), input_mask: (B, T_in, max_N) bool.
        Returns (B, output_size).
  AD-2: Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding=1)
        applied per-timestep via reshape to (B*T_in, 1, max_N).
        LSTM input_size = conv_channels * max_N + context_size.
  AD-3: inp_masked = inp * input_mask.float() BEFORE the Conv1d spatial layer.
        Absent slots (mask=False) are zeroed at the model boundary.
        No post-conv mask applied — LSTM sees the full spatial feature vector.
  AD-7: PyTorch default weight initialization (Kaiming uniform for Conv1d,
        uniform for LSTM). No custom init to ensure fair comparison with HeadwayLSTM.

Duck-type dispatch:
  self.spatial = True  ← class attribute checked by train.py (AD-4).

INV-NO-COMPILE: torch.compile() is NOT used (Kaggle CUDA compatibility).
INV-NO-TB: No TensorBoard, MLflow, or Optuna imports.

ACs covered: W1-C2 spatial-conv-lstm-model scenarios.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpatialConvLSTM(nn.Module):
    """1D spatial conv encoder followed by an LSTM temporal encoder.

    The model applies a 1D convolution over the spatial (N) dimension at
    each timestep, then feeds the resulting spatial features concatenated
    with context through an LSTM. Only the last hidden state is projected
    to the output.

    Parameters
    ----------
    max_N:
        Number of spatial positions (bus slots). The Conv1d treats this as
        the spatial sequence length.
    conv_channels:
        Number of output channels for the Conv1d layer. Determines LSTM
        input_size = conv_channels * max_N + context_size.
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
    context_size:
        Dimensionality of the context vector at each timestep (default 5).
        Context is: hour_sin, hour_cos, dow_sin, dow_cos, atypical.
    """

    # Duck-type flag: train.py checks hasattr(model, 'spatial') and model.spatial
    # to dispatch the 3-argument forward call (AD-4).
    spatial: bool = True

    def __init__(
        self,
        max_N: int,
        conv_channels: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        context_size: int = 5,
    ) -> None:
        super().__init__()

        self.max_N = max_N
        self.conv_channels = conv_channels
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.context_size = context_size

        # 1D spatial conv: treats the N dimension as a 1-channel spatial sequence.
        # kernel_size=3, padding=1 → same spatial length; 1-hop message passing (AD-2).
        # in_channels=1 (single feature per position), out_channels=conv_channels.
        self.spatial_conv = nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
        )

        # LSTM input dimension: flattened conv output + context vector (AD-2).
        lstm_input_size = conv_channels * max_N + context_size

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Linear head: last hidden state → spatial output predictions.
        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        inp: torch.Tensor,
        ctx: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Spatial conv + LSTM forward pass.

        Parameters
        ----------
        inp:
            (B, T_in, max_N) float32 — z-scored headway values; 0.0 for absent slots.
        ctx:
            (B, T_in, context_size) float32 — cyclical time + atypical flag.
        input_mask:
            (B, T_in, max_N) bool — True where slot is present and non-null.

        Returns
        -------
        (B, output_size) float32 — predicted next headway vector (z-scored).
        Uses only the last hidden state (mirrors HeadwayLSTM AD-2).
        """
        B, T_in, max_N = inp.shape

        # --- Step 1: Pre-conv masking (AD-3) ---
        # Zero absent slots BEFORE the convolution sees them.
        inp_masked = inp * input_mask.float()  # (B, T_in, max_N)

        # --- Step 2: Reshape for Conv1d ---
        # Conv1d expects (N, C_in, L); treat each (batch, timestep) independently.
        # Merge B and T_in into batch dimension, treat max_N as the 1D spatial length.
        inp_for_conv = inp_masked.reshape(B * T_in, 1, max_N)  # (B*T_in, 1, max_N)

        # --- Step 3: Apply spatial conv ---
        # Output: (B*T_in, conv_channels, max_N) — same spatial length (padding=1).
        conv_out = self.spatial_conv(inp_for_conv)  # (B*T_in, conv_channels, max_N)

        # --- Step 4: Reshape spatial features back ---
        # Flatten the conv output per timestep: conv_channels * max_N features.
        # Then restore the time dimension.
        spatial_features = conv_out.reshape(B, T_in, self.conv_channels * max_N)
        # (B, T_in, conv_channels * max_N)

        # --- Step 5: Concatenate with context ---
        # Context carries temporal signals; spatial features carry neighbourhood info.
        lstm_input = torch.cat([spatial_features, ctx], dim=-1)
        # (B, T_in, conv_channels * max_N + context_size)

        # --- Step 6: LSTM over time dimension ---
        # batch_first=True → lstm_out: (B, T_in, hidden_size), h_n: (num_layers, B, hidden_size)
        _lstm_out, (h_n, _c_n) = self.lstm(lstm_input)

        # Take the topmost layer's last hidden state.
        last_hidden = h_n[-1]  # (B, hidden_size)

        # --- Step 7: Linear projection to output space ---
        return self.head(last_hidden)  # (B, output_size)
