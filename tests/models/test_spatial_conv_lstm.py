"""Tests for SpatialConvLSTM — W1-C1 RED.

Strict TDD: these tests are written BEFORE the implementation exists.
They MUST fail (ImportError) initially and pass after
src/models/spatial_conv_lstm.py is created.

Design refs:
  - AD-1: forward(inp, ctx, input_mask) — 3 separate tensors.
  - AD-2: Conv1d(1, conv_channels, 3, padding=1) per timestep;
          LSTM input_size = conv_channels * max_N + context_size.
  - AD-3: inp_masked = inp * input_mask.float() BEFORE Conv1d.
  - AD-7: PyTorch default weight init.
  - Duck-type dispatch flag: model.spatial is True.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(
    max_N: int = 4,
    conv_channels: int = 8,
    hidden_size: int = 32,
    output_size: int | None = None,
    num_layers: int = 1,
    dropout: float = 0.0,
    context_size: int = 5,
):
    """Construct a SpatialConvLSTM with given parameters."""
    from src.models.spatial_conv_lstm import SpatialConvLSTM
    return SpatialConvLSTM(
        max_N=max_N,
        conv_channels=conv_channels,
        hidden_size=hidden_size,
        output_size=output_size if output_size is not None else max_N,
        num_layers=num_layers,
        dropout=dropout,
        context_size=context_size,
    )


def _make_batch(B: int = 2, T: int = 12, max_N: int = 4, all_valid: bool = True):
    """Create synthetic (inp, ctx, input_mask) tensors."""
    inp = torch.randn(B, T, max_N)
    ctx = torch.randn(B, T, 5)
    if all_valid:
        input_mask = torch.ones(B, T, max_N, dtype=torch.bool)
    else:
        input_mask = torch.zeros(B, T, max_N, dtype=torch.bool)
    return inp, ctx, input_mask


# ---------------------------------------------------------------------------
# TestSpatialConvLSTM
# ---------------------------------------------------------------------------

class TestSpatialConvLSTM:
    """Unit tests for SpatialConvLSTM (W1-C1 RED)."""

    def test_model_is_nn_module(self):
        """SpatialConvLSTM is a torch.nn.Module."""
        model = _make_model()
        assert isinstance(model, nn.Module)

    def test_spatial_flag_true(self):
        """model.spatial is True — duck-type dispatch flag for train.py."""
        model = _make_model()
        assert model.spatial is True

    def test_forward_output_shape(self):
        """AD-1: forward(inp, ctx, mask) with B=2, T=12, max_N=4 → output (2, 4)."""
        model = _make_model(max_N=4, conv_channels=8, hidden_size=32, output_size=4)
        inp, ctx, mask = _make_batch(B=2, T=12, max_N=4)
        with torch.no_grad():
            out = model(inp, ctx, mask)
        assert out.shape == (2, 4)

    def test_forward_batch_size_one(self):
        """B=1 produces (1, max_N) output with no NaN."""
        model = _make_model(max_N=4, output_size=4)
        inp, ctx, mask = _make_batch(B=1, T=12, max_N=4)
        with torch.no_grad():
            out = model(inp, ctx, mask)
        assert out.shape == (1, 4)
        assert not torch.isnan(out).any()

    def test_forward_dtype_float32(self):
        """Output dtype is torch.float32."""
        model = _make_model()
        inp, ctx, mask = _make_batch()
        with torch.no_grad():
            out = model(inp, ctx, mask)
        assert out.dtype == torch.float32

    def test_mask_zeros_absent_slots(self):
        """AD-3: changing inp[0,:,2] when mask[0,:,2]=False does NOT change output.

        Pre-conv masking: inp_masked = inp * input_mask.float()
        Absent slots are zeroed before the convolution, so their content is irrelevant.
        """
        model = _make_model(max_N=4, output_size=4)
        model.eval()

        B, T, max_N = 2, 12, 4
        inp = torch.randn(B, T, max_N)
        ctx = torch.randn(B, T, 5)
        # Mask slot 2 for sample 0 as absent across all timesteps.
        mask = torch.ones(B, T, max_N, dtype=torch.bool)
        mask[0, :, 2] = False

        with torch.no_grad():
            out_original = model(inp, ctx, mask)

        # Mutate the masked slot to arbitrary large values.
        inp_mutated = inp.clone()
        inp_mutated[0, :, 2] = 999.0

        with torch.no_grad():
            out_mutated = model(inp_mutated, ctx, mask)

        assert torch.allclose(out_original, out_mutated), (
            "Changing masked (absent) slot values should NOT change model output."
        )

    def test_fully_valid_mask_identity(self):
        """AD-3: all-True mask means inp * mask.float() == inp (no data lost)."""
        B, T, max_N = 2, 12, 4
        inp = torch.randn(B, T, max_N)
        mask = torch.ones(B, T, max_N, dtype=torch.bool)
        # Verify masking operation directly.
        masked = inp * mask.float()
        assert torch.allclose(inp, masked)

    def test_conv_channels_1_shape(self):
        """AD-2: conv_channels=1 → LSTM input_size = 1*max_N + context_size = 4+5=9."""
        max_N, context_size = 4, 5
        model = _make_model(max_N=max_N, conv_channels=1, context_size=context_size)
        expected_lstm_input = 1 * max_N + context_size  # 9
        assert model.lstm.input_size == expected_lstm_input
        # Also check forward shape.
        inp, ctx, mask = _make_batch(B=2, T=12, max_N=max_N)
        with torch.no_grad():
            out = model(inp, ctx, mask)
        assert out.shape == (2, max_N)

    def test_conv_channels_8_shape(self):
        """AD-2: conv_channels=8 → LSTM input_size = 8*max_N + context_size = 32+5=37."""
        max_N, context_size = 4, 5
        model = _make_model(max_N=max_N, conv_channels=8, context_size=context_size)
        expected_lstm_input = 8 * max_N + context_size  # 37
        assert model.lstm.input_size == expected_lstm_input
        # Also check forward shape.
        inp, ctx, mask = _make_batch(B=2, T=12, max_N=max_N)
        with torch.no_grad():
            out = model(inp, ctx, mask)
        assert out.shape == (2, max_N)

    def test_num_layers_2_correct_shape(self):
        """Two-layer variant (num_layers=2) produces (B, max_N) output without error."""
        model = _make_model(
            max_N=4,
            conv_channels=8,
            hidden_size=32,
            output_size=4,
            num_layers=2,
            dropout=0.2,
        )
        inp, ctx, mask = _make_batch(B=2, T=12, max_N=4)
        with torch.no_grad():
            out = model(inp, ctx, mask)
        assert out.shape == (2, 4)

    def test_gradient_flows_end_to_end(self):
        """loss.backward() populates non-None finite grads on conv and lstm params."""
        from src.models.lstm import masked_mse_loss

        model = _make_model(max_N=4, conv_channels=8, hidden_size=32, output_size=4)
        model.train()

        B, T, max_N = 2, 12, 4
        inp = torch.randn(B, T, max_N)
        ctx = torch.randn(B, T, 5)
        mask = torch.ones(B, T, max_N, dtype=torch.bool)
        target = torch.randn(B, max_N)
        target_mask = torch.ones(B, max_N, dtype=torch.bool)

        pred = model(inp, ctx, mask)
        loss = masked_mse_loss(pred, target, target_mask)
        loss.backward()

        # Every named parameter must have a non-None, finite gradient.
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient is None for {name}"
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient in {name}"
            )

    def test_dropout_zero_eval_deterministic(self):
        """dropout=0.0 in .eval() mode → identical output for same inputs."""
        model = _make_model(max_N=4, conv_channels=8, dropout=0.0)
        model.eval()

        inp, ctx, mask = _make_batch(B=2, T=12, max_N=4)
        with torch.no_grad():
            out1 = model(inp, ctx, mask)
            out2 = model(inp, ctx, mask)

        assert torch.allclose(out1, out2), (
            "Same inputs in eval mode must produce identical outputs."
        )
