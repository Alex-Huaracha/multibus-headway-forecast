"""SpatialTransformer — multi-head self-attention spatial encoder + LSTM temporal encoder.

Architecture decisions applied (from design doc):
  AD-1: Forward signature is (inp, ctx, input_mask) — 3 separate tensors.
        inp: (B, T_in, max_N), ctx: (B, T_in, context_size), input_mask: (B, T_in, max_N) bool.
        Returns (B, output_size).
  AD-2: Per-timestep MHA via B*T_in batched reshape (no Python loop over T_in).
        proj_in: Linear(1, d_model) per position.
        MHA: MultiheadAttention(d_model, nhead, batch_first=True) self-attention.
        LSTM input_size = max_N + context_size (SAME as HeadwayLSTM — AD-3).
  AD-3: proj_out: Linear(d_model, 1) collapses d_model→1 per position after MHA.
        This keeps LSTM input width minimal: max_N + context_size (not d_model*max_N).
        Residual + LayerNorm before proj_out for training stability.
  AD-4: Mask polarity inversion: key_padding_mask = ~input_mask.
        Dataset convention: True=valid; PyTorch MHA convention: True=IGNORE.
  AD-5: NaN guard for fully-masked snapshots.
        When all keys in a B*T_in row are masked, MHA emits NaN.
        Guard: force first key valid for those rows in safe_kpm; zero their attn_out after MHA.

Duck-type dispatch:
  spatial: bool = True  ← class attribute checked by train.py (AD-4 duck-type).

INV-NO-COMPILE: torch.compile() is NOT used (Kaggle CUDA compatibility).
INV-NO-TB: No TensorBoard, MLflow, or Optuna imports.

ACs covered: W1-C2 spatial-transformer-model scenarios.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpatialTransformer(nn.Module):
    """Multi-head self-attention spatial encoder + LSTM temporal encoder.

    Applies per-timestep MHA over the N spatial positions (bus slots),
    then feeds the attended spatial features concatenated with context
    through an LSTM. Only the last hidden state is projected to output.

    The spatial attention step uses a single batched MHA call over the
    B*T_in dimension — no Python loop over T_in (D1 runtime constraint).

    Parameters
    ----------
    max_N:
        Number of spatial positions (bus slots).
    nhead:
        Number of attention heads. Must satisfy d_model % nhead == 0.
    d_model:
        Attention embedding dimension per position.
    hidden_size:
        Number of LSTM hidden units.
    output_size:
        Number of output positions (= max_N). The linear head maps
        hidden_size → output_size.
    num_layers:
        Number of stacked LSTM layers (default 1).
    dropout:
        Dropout probability in MHA and between LSTM layers (default 0.0).
        LSTM dropout is ignored when num_layers == 1 (PyTorch behaviour).
    context_size:
        Dimensionality of the context vector at each timestep (default 5).
        Context is: hour_sin, hour_cos, dow_sin, dow_cos, atypical.
    """

    # Duck-type flag: train.py checks hasattr(model, 'spatial') and model.spatial
    # to dispatch the 3-argument forward call. Shared with SpatialConvLSTM (AD-6).
    spatial: bool = True

    def __init__(
        self,
        max_N: int,
        nhead: int,
        d_model: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        context_size: int = 5,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
                "MultiheadAttention requires d_model % nhead == 0."
            )

        self.max_N = max_N
        self.nhead = nhead
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.context_size = context_size

        # --- Spatial encoder layers ---

        # Project each position's scalar headway to d_model dimensions.
        # Input: (B*T_in, max_N, 1) → Output: (B*T_in, max_N, d_model)
        self.proj_in = nn.Linear(1, d_model)

        # Multi-head self-attention over N positions per timestep.
        # batch_first=True: input/output shape is (batch, seq, d_model).
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention layer norm (residual path: proj_in + attn_out).
        self.norm = nn.LayerNorm(d_model)

        # Project each attended position back to a scalar (AD-3).
        # Output: (B*T_in, max_N, 1) → squeeze → (B*T_in, max_N)
        self.proj_out = nn.Linear(d_model, 1)

        # --- Temporal encoder ---

        # LSTM input width = max_N + context_size (AD-3, same as HeadwayLSTM).
        lstm_input_size = max_N + context_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Linear head: last LSTM hidden state → spatial output predictions.
        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        inp: torch.Tensor,
        ctx: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        """MHA spatial encoder + LSTM temporal forward pass.

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
        """
        B, T_in, max_N = inp.shape

        # --- Step 1: Pre-attention masking ---
        # Zero absent slots before projection (clean input to proj_in).
        inp_masked = inp * input_mask.float()  # (B, T_in, max_N)

        # --- Step 2: Reshape for batched MHA ---
        # Fold (B, T_in) into a single batch dimension.
        # Then add feature dim for proj_in.
        x = inp_masked.reshape(B * T_in, max_N, 1)  # (B*T_in, max_N, 1)
        x = self.proj_in(x)                          # (B*T_in, max_N, d_model)

        # --- Step 3: Mask polarity inversion (AD-4) ---
        # Dataset: True=valid. PyTorch MHA key_padding_mask: True=IGNORE.
        kpm = ~input_mask.reshape(B * T_in, max_N)   # (B*T_in, max_N) bool

        # --- Step 4: NaN guard for fully-masked snapshots (AD-5) ---
        # MHA emits NaN when an entire key row is masked (all True in kpm).
        # Detection: which rows have ALL positions masked?
        all_masked = kpm.all(dim=1)  # (B*T_in,) bool

        # Build safe_kpm: for fully-masked rows, force the first position valid
        # so MHA can compute a valid (but meaningless) attention score.
        safe_kpm = kpm.clone()
        if all_masked.any():
            safe_kpm[all_masked, 0] = False  # first key is "visible" for safety

        # --- Step 5: Self-attention ---
        # Q = K = V = x (self-attention over N positions per timestep).
        attn_out, _ = self.attn(x, x, x, key_padding_mask=safe_kpm)
        # attn_out: (B*T_in, max_N, d_model)

        # Zero out MHA output for rows that were fully masked (AD-5).
        # The guard above let MHA run, but its output is meaningless there.
        if all_masked.any():
            attn_out[all_masked] = 0.0

        # --- Step 6: Residual + LayerNorm + project back to scalar (AD-3) ---
        x = self.norm(x + attn_out)                      # (B*T_in, max_N, d_model)
        x = self.proj_out(x).squeeze(-1)                 # (B*T_in, max_N)

        # Ensure fully-masked snapshots remain zero after proj_out.
        if all_masked.any():
            x[all_masked] = 0.0

        # --- Step 7: Restore time dimension ---
        spatial = x.reshape(B, T_in, max_N)              # (B, T_in, max_N)

        # --- Step 8: Concatenate with context ---
        lstm_input = torch.cat([spatial, ctx], dim=-1)   # (B, T_in, max_N + context_size)

        # --- Step 9: LSTM over time dimension ---
        # h_n: (num_layers, B, hidden_size)
        _lstm_out, (h_n, _c_n) = self.lstm(lstm_input)

        # Take the topmost layer's last hidden state.
        last_hidden = h_n[-1]  # (B, hidden_size)

        # --- Step 10: Linear projection to output space ---
        return self.head(last_hidden)  # (B, output_size)
