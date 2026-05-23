"""Tests for HeadwayLSTM and masked_mse_loss — W1 RED.

Strict TDD: these tests are written BEFORE the implementation exists.
They MUST fail initially and pass after src/models/lstm.py is created.

ACs covered: AC-MODEL-1..5, AC-LOSS-1..5.

Design refs:
  - design AD-1: input = flat concat (max_N + context_size); caller concatenates.
  - design AD-2: last hidden state → Linear → (B, output_size).
  - design AD-3: mask applied only at loss; model always produces output for all positions.
  - design §3.2: forward(self, x: Tensor) -> Tensor; x: (B, T_in, input_size).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(
    input_size: int = 7,
    hidden_size: int = 32,
    output_size: int = 4,
    num_layers: int = 1,
    dropout: float = 0.0,
):
    """Construct a HeadwayLSTM with given parameters."""
    from src.models.lstm import HeadwayLSTM
    return HeadwayLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# TestHeadwayLSTM
# ---------------------------------------------------------------------------

class TestHeadwayLSTM:
    """Tests for HeadwayLSTM forward pass and structure (AC-MODEL-1..5)."""

    def test_model_is_nn_module(self):
        """AC-MODEL-1: HeadwayLSTM is a torch.nn.Module."""
        model = _make_model()
        assert isinstance(model, nn.Module)

    def test_forward_output_shape(self):
        """AC-MODEL-1, AC-MODEL-2: forward(x) with (B=2, T=12, input_size=7) → (2, 4).

        Tasks spec: HeadwayLSTM(input_size=7, hidden_size=32, output_size=4)
        with input (2, 12, 7) → output (2, 4).
        """
        model = _make_model(input_size=7, hidden_size=32, output_size=4)
        x = torch.randn(2, 12, 7)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4)

    def test_forward_batch_size_one(self):
        """AC-MODEL-2: single-sample batch (1, 12, 7) → (1, 4)."""
        model = _make_model(input_size=7, hidden_size=32, output_size=4)
        x = torch.randn(1, 12, 7)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4)

    def test_forward_dtype_float32(self):
        """AC-MODEL-3: output dtype is torch.float32."""
        model = _make_model()
        x = torch.randn(2, 12, 7)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_num_layers_2(self):
        """AC-MODEL-4: two-layer LSTM with dropout=0.2 produces correct output shape."""
        model = _make_model(
            input_size=7,
            hidden_size=32,
            output_size=4,
            num_layers=2,
            dropout=0.2,
        )
        x = torch.randn(2, 12, 7)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4)

    def test_dropout_zero_eval_deterministic(self):
        """AC-MODEL-5: dropout=0.0 in eval mode → identical output for same input."""
        model = _make_model(input_size=7, hidden_size=32, output_size=4, dropout=0.0)
        model.eval()
        x = torch.randn(2, 12, 7)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_model_parameter_count(self):
        """AC-MODEL-1: Parameter count matches LSTM formula.

        Single-layer LSTM: 4 * (input_size + hidden_size + 1) * hidden_size
        Linear head: hidden_size * output_size + output_size
        """
        input_size = 7
        hidden_size = 16
        output_size = 4
        model = _make_model(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=1,
            dropout=0.0,
        )
        total_params = sum(p.numel() for p in model.parameters())
        # LSTM params = 4 * hidden * (input + hidden) + 4 * hidden (biases)
        lstm_params = 4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size
        # Linear params
        linear_params = hidden_size * output_size + output_size
        expected = lstm_params + linear_params
        assert total_params == expected, (
            f"Expected {expected} parameters, got {total_params}"
        )

    def test_forward_output_independent_of_mask(self):
        """AC-MODEL-2 (mask handling): model produces output for ALL positions.

        The model does not apply any mask internally; it always returns
        shape (B, output_size) regardless of which positions are valid.
        The mask is only used by masked_mse_loss, not inside forward.
        """
        model = _make_model(input_size=7, hidden_size=32, output_size=4)
        x = torch.randn(2, 12, 7)
        with torch.no_grad():
            out = model(x)
        # All output positions are produced (no NaN from masking)
        assert out.shape == (2, 4)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# TestMaskedMSELoss
# ---------------------------------------------------------------------------

class TestMaskedMSELoss:
    """Tests for masked_mse_loss function (AC-LOSS-1..5)."""

    def test_masked_mse_all_valid(self):
        """AC-LOSS-1: all-True mask equals F.mse_loss mean within 1e-6."""
        from src.models.lstm import masked_mse_loss

        B, N = 4, 6
        pred = torch.randn(B, N)
        target = torch.randn(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)

        loss = masked_mse_loss(pred, target, mask)
        expected = F.mse_loss(pred, target, reduction="mean")
        assert abs(loss.item() - expected.item()) < 1e-6

    def test_masked_mse_single_position(self):
        """AC-LOSS-2: mask with exactly one True position at [0, 0]."""
        from src.models.lstm import masked_mse_loss

        B, N = 3, 5
        pred = torch.zeros(B, N)
        target = torch.zeros(B, N)
        pred[0, 0] = 2.0
        target[0, 0] = 1.0

        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[0, 0] = True

        loss = masked_mse_loss(pred, target, mask)
        # (2.0 - 1.0)^2 / 1 = 1.0
        assert abs(loss.item() - 1.0) < 1e-6

    def test_masked_mse_all_masked(self):
        """AC-LOSS-3: all-False mask → 0.0 (clamp(min=1) prevents zero-division)."""
        from src.models.lstm import masked_mse_loss

        B, N = 4, 6
        pred = torch.ones(B, N)
        target = torch.zeros(B, N)
        mask = torch.zeros(B, N, dtype=torch.bool)

        loss = masked_mse_loss(pred, target, mask)
        assert loss.item() == 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_masked_mse_returns_scalar_tensor(self):
        """AC-LOSS-4: return is a 0-dim tensor; .backward() works."""
        from src.models.lstm import masked_mse_loss

        B, N = 2, 4
        pred = torch.randn(B, N, requires_grad=True)
        target = torch.randn(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)

        loss = masked_mse_loss(pred, target, mask)
        assert loss.dim() == 0, "Loss must be a scalar (0-dim) tensor"
        loss.backward()
        assert pred.grad is not None

    def test_masked_mse_masked_position_ignored(self):
        """AC-LOSS-5: changing a masked position value does not change the loss."""
        from src.models.lstm import masked_mse_loss

        B, N = 2, 4
        pred_a = torch.randn(B, N)
        pred_b = pred_a.clone()
        target = torch.randn(B, N)

        mask = torch.ones(B, N, dtype=torch.bool)
        mask[0, 1] = False  # position [0,1] is masked out

        # Mutate the masked position
        pred_b[0, 1] = 999.0

        loss_a = masked_mse_loss(pred_a, target, mask)
        loss_b = masked_mse_loss(pred_b, target, mask)

        assert abs(loss_a.item() - loss_b.item()) < 1e-6

    def test_masked_mse_known_values(self):
        """AC-LOSS-1, AC-LOSS-2: hand-computed verification with known pred/target/mask.

        pred   = [[1.0, 2.0, 3.0]]
        target = [[0.0, 0.0, 0.0]]
        mask   = [[True, True, False]]

        Valid positions: 0 and 1.
        SE: (1-0)^2 + (2-0)^2 = 1 + 4 = 5
        Count: 2
        Loss: 5 / 2 = 2.5
        """
        from src.models.lstm import masked_mse_loss

        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False]])

        loss = masked_mse_loss(pred, target, mask)
        assert abs(loss.item() - 2.5) < 1e-6
