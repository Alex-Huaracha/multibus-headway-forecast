"""Tests for SpatialTransformer — W1-C1 RED.

Strict TDD: these tests are written BEFORE the implementation exists.
They MUST fail (ImportError) initially and pass after
src/models/spatial_transformer.py is created.

Design refs:
  - AD-1: forward(inp, ctx, input_mask) — 3 separate tensors.
  - AD-2: MHA self-attention over max_N positions, batched B*T_in reshape.
  - AD-3: proj_out Linear(d_model, 1) collapses d_model→1 per position so
          LSTM input_size = max_N + context_size (same as HeadwayLSTM).
  - AD-4: key_padding_mask = ~input_mask (polarity inversion).
  - AD-5: NaN guard for fully-masked snapshots.
  - Duck-type dispatch flag: model.spatial is True (class attribute).

Grid decision (obs #417):
  TRANSFORMER_GRID = nhead{1,2} × d_model{16,32} × hidden{32,64}
                     × dropout{0.0,0.2} × lr{1e-3,5e-4} → 32 configs
  d_model values are {16, 32} (NOT {8, 16}).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(
    max_N: int = 10,
    nhead: int = 2,
    d_model: int = 16,
    hidden_size: int = 32,
    output_size: int | None = None,
    num_layers: int = 1,
    dropout: float = 0.0,
    context_size: int = 5,
):
    """Construct a SpatialTransformer with given parameters."""
    from src.models.spatial_transformer import SpatialTransformer
    return SpatialTransformer(
        max_N=max_N,
        nhead=nhead,
        d_model=d_model,
        hidden_size=hidden_size,
        output_size=output_size if output_size is not None else max_N,
        num_layers=num_layers,
        dropout=dropout,
        context_size=context_size,
    )


def _make_batch(B: int = 2, T: int = 12, max_N: int = 10, all_valid: bool = True):
    """Create synthetic (inp, ctx, input_mask) tensors."""
    inp = torch.randn(B, T, max_N)
    ctx = torch.randn(B, T, 5)
    if all_valid:
        input_mask = torch.ones(B, T, max_N, dtype=torch.bool)
    else:
        input_mask = torch.zeros(B, T, max_N, dtype=torch.bool)
    return inp, ctx, input_mask


# ---------------------------------------------------------------------------
# TestSpatialTransformer
# ---------------------------------------------------------------------------

class TestSpatialTransformer:
    """Unit tests for SpatialTransformer (W1-C1 RED)."""

    def test_model_is_nn_module(self):
        """SpatialTransformer is a torch.nn.Module."""
        model = _make_model()
        assert isinstance(model, nn.Module)

    def test_spatial_flag_true(self):
        """model.spatial is True — duck-type dispatch flag for train.py (AD-4)."""
        model = _make_model()
        assert hasattr(model, "spatial") and model.spatial is True

    def test_forward_output_shape(self):
        """AD-1: forward(inp, ctx, mask) with B=2, T=12, max_N=10 → output (2, 10)."""
        model = _make_model(max_N=10, nhead=2, d_model=16, hidden_size=32, output_size=10)
        inp, ctx, mask = _make_batch(B=2, T=12, max_N=10)
        with torch.no_grad():
            out = model(inp, ctx, mask)
        assert out.shape == (2, 10)

    def test_forward_batch_size_one(self):
        """B=1 produces (1, max_N) output with no NaN."""
        model = _make_model(max_N=10, output_size=10)
        inp, ctx, mask = _make_batch(B=1, T=12, max_N=10)
        with torch.no_grad():
            out = model(inp, ctx, mask)
        assert out.shape == (1, 10)
        assert not torch.isnan(out).any()

    def test_forward_dtype_float32(self):
        """Output dtype is torch.float32."""
        model = _make_model()
        inp, ctx, mask = _make_batch()
        with torch.no_grad():
            out = model(inp, ctx, mask)
        assert out.dtype == torch.float32

    def test_mask_ignores_absent_slots(self):
        """AD-4: changing inp[0,:,3] when mask[0,:,3]=False does NOT change output.

        Polarity inversion: kpm = ~input_mask, so absent slots are IGNORED by MHA.
        Their content is irrelevant — changing an absent slot must not change the output.
        """
        model = _make_model(max_N=10, output_size=10)
        model.eval()

        B, T, max_N = 2, 12, 10
        inp = torch.randn(B, T, max_N)
        ctx = torch.randn(B, T, 5)
        # Mask slot 3 for sample 0 as absent across all timesteps.
        mask = torch.ones(B, T, max_N, dtype=torch.bool)
        mask[0, :, 3] = False

        with torch.no_grad():
            out_original = model(inp, ctx, mask)

        # Mutate the masked slot to arbitrary large values.
        inp_mutated = inp.clone()
        inp_mutated[0, :, 3] = 999.0

        with torch.no_grad():
            out_mutated = model(inp_mutated, ctx, mask)

        assert torch.allclose(out_original, out_mutated), (
            "Changing masked (absent) slot values should NOT change model output."
        )

    def test_all_masked_snapshot_no_nan(self):
        """AD-5: input_mask[0, 2, :] all-False (fully-absent snapshot) → no NaN in output.

        Uses a REAL all-False snapshot (not near-full) to trigger the guard path.
        The NaN guard must prevent PyTorch MHA from emitting NaN for fully-masked rows.
        """
        model = _make_model(max_N=10, output_size=10)
        model.eval()

        B, T, max_N = 2, 12, 10
        inp = torch.randn(B, T, max_N)
        ctx = torch.randn(B, T, 5)
        # All-valid mask except for one fully-absent snapshot.
        mask = torch.ones(B, T, max_N, dtype=torch.bool)
        mask[0, 2, :] = False  # snapshot at t=2 for sample 0: ALL slots absent

        with torch.no_grad():
            out = model(inp, ctx, mask)

        assert torch.isfinite(out).all(), (
            "Forward with a fully-absent snapshot must not produce NaN or Inf. "
            "The NaN guard (AD-5) must zero out MHA output for fully-masked rows."
        )

    def test_partial_mask_valid_slot_finite(self):
        """AD-5: only one slot valid in a snapshot → output is finite.

        This is NOT the all-masked case — one slot IS valid so the guard
        should not fire, but output must still be finite.
        """
        model = _make_model(max_N=10, output_size=10)
        model.eval()

        B, T, max_N = 2, 12, 10
        inp = torch.randn(B, T, max_N)
        ctx = torch.randn(B, T, 5)
        # Only slot 0 is valid for sample 0 at t=5.
        mask = torch.ones(B, T, max_N, dtype=torch.bool)
        mask[0, 5, 1:] = False  # only slot 0 valid at t=5 for sample 0

        with torch.no_grad():
            out = model(inp, ctx, mask)

        assert torch.isfinite(out).all(), (
            "Output must be finite even when only one slot is valid in a snapshot."
        )

    def test_d_model_not_divisible_by_nhead_raises(self):
        """Constructor must raise ValueError when d_model % nhead != 0 (MHA constraint)."""
        with pytest.raises(ValueError, match=r"d_model.*nhead|nhead.*d_model"):
            _make_model(nhead=3, d_model=16)  # 16 % 3 != 0

    def test_nhead_1_shape(self):
        """nhead=1, d_model=16 → output shape is (B, max_N) with no error."""
        model = _make_model(nhead=1, d_model=16, max_N=10, output_size=10)
        inp, ctx, mask = _make_batch(B=2, T=12, max_N=10)
        with torch.no_grad():
            out = model(inp, ctx, mask)
        assert out.shape == (2, 10)

    def test_lstm_input_width_is_max_N_plus_5(self):
        """AD-3: proj_out collapses d_model→1 per position so lstm.input_size == max_N + 5.

        This is DIFFERENT from the conv-lstm where LSTM width = conv_channels*max_N + 5.
        Here: proj_out Linear(d_model, 1) → each position gets ONE feature,
        so LSTM input_size = max_N + context_size = max_N + 5.
        """
        max_N, context_size = 10, 5
        model = _make_model(
            max_N=max_N, nhead=2, d_model=16, hidden_size=32, context_size=context_size
        )
        assert model.lstm.input_size == max_N + context_size, (
            f"LSTM input_size must be max_N + context_size = {max_N + context_size}, "
            f"got {model.lstm.input_size}"
        )

    def test_gradient_flows_through_all_params(self):
        """loss.backward() populates non-None finite grads on all leaf params (attn + lstm + head)."""
        from src.models.lstm import masked_mse_loss

        model = _make_model(max_N=10, nhead=2, d_model=16, hidden_size=32, output_size=10)
        model.train()

        B, T, max_N = 2, 12, 10
        inp = torch.randn(B, T, max_N)
        ctx = torch.randn(B, T, 5)
        mask = torch.ones(B, T, max_N, dtype=torch.bool)
        target = torch.randn(B, max_N)
        target_mask = torch.ones(B, max_N, dtype=torch.bool)

        pred = model(inp, ctx, mask)
        loss = masked_mse_loss(pred, target, target_mask)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None for {name}"
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in {name}"
                )

    def test_dropout_zero_eval_deterministic(self):
        """dropout=0.0 in .eval() mode → identical output for same inputs."""
        model = _make_model(max_N=10, nhead=2, d_model=16, dropout=0.0)
        model.eval()

        inp, ctx, mask = _make_batch(B=2, T=12, max_N=10)
        with torch.no_grad():
            out1 = model(inp, ctx, mask)
            out2 = model(inp, ctx, mask)

        assert torch.allclose(out1, out2), (
            "Same inputs in eval mode with dropout=0.0 must produce identical outputs."
        )
