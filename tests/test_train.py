"""Tests for training primitives — Wave 2 (RED phase).

Covers:
    AC-TRAIN: TrainConfig dataclass, set_seed, train_one_epoch,
              evaluate_epoch, EarlyStopping.

Test runner: uv run pytest tests/test_train.py -v
"""
from __future__ import annotations

import copy

import pytest
import torch

from src.train import (
    EarlyStopping,
    TrainConfig,
    evaluate_epoch,
    set_seed,
    train_one_epoch,
)
from src.models.lstm import HeadwayLSTM


# ---------------------------------------------------------------------------
# Synthetic dataloader helper
# ---------------------------------------------------------------------------

def _make_synthetic_dataloader(
    max_N: int = 3,
    context_dim: int = 5,
    seq_len: int = 6,
    batch_size: int = 4,
    n_batches: int = 2,
) -> list[dict[str, torch.Tensor]]:
    """Create a minimal list of dict batches matching HeadwayDataset contract.

    Each batch has keys: input (B, T_in, max_N), context (B, T_in, 5),
    target (B, T_out, max_N), target_mask (B, T_out, max_N).
    train_one_epoch / evaluate_epoch must concatenate input + context before
    passing to the model (AD-1).
    """
    batches = []
    for _ in range(n_batches):
        inp = torch.randn(batch_size, seq_len, max_N)
        ctx = torch.randn(batch_size, seq_len, context_dim)
        target = torch.randn(batch_size, 1, max_N)
        mask = torch.ones(batch_size, 1, max_N, dtype=torch.bool)
        batches.append({"input": inp, "context": ctx, "target": target, "target_mask": mask})
    return batches


def _make_model(max_N: int = 3, context_dim: int = 5, hidden_size: int = 16) -> HeadwayLSTM:
    """Build a tiny HeadwayLSTM for testing."""
    return HeadwayLSTM(
        input_size=max_N + context_dim,
        hidden_size=hidden_size,
        output_size=max_N,
        num_layers=1,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# TrainConfig tests
# ---------------------------------------------------------------------------

class TestTrainConfig:
    def test_train_config_dataclass_defaults(self) -> None:
        """TrainConfig must expose all required fields with correct defaults."""
        cfg = TrainConfig(
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            lr=1e-3,
        )
        assert cfg.hidden_size == 64
        assert cfg.num_layers == 1
        assert cfg.dropout == 0.0
        assert cfg.lr == 1e-3
        # Defaults
        assert cfg.batch_size == 32
        assert cfg.max_epochs == 50
        assert cfg.patience == 10
        assert cfg.seed == 42


# ---------------------------------------------------------------------------
# set_seed tests
# ---------------------------------------------------------------------------

class TestSetSeed:
    def test_set_seed_same_seed_produces_identical_tensor(self) -> None:
        """Two set_seed calls with the same seed must produce identical random tensors."""
        set_seed(123)
        t1 = torch.randn(10)
        set_seed(123)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2), "Same seed must produce identical tensors"

    def test_set_seed_same_seed_identical_model_weights(self) -> None:
        """Two HeadwayLSTM instances initialized after same seed must have identical weights."""
        set_seed(7)
        m1 = _make_model()
        set_seed(7)
        m2 = _make_model()
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2), "Same seed must produce identical weights"


# ---------------------------------------------------------------------------
# train_one_epoch tests
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:
    def test_train_one_epoch_returns_float(self) -> None:
        """train_one_epoch must return a Python float."""
        model = _make_model()
        loader = _make_synthetic_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = train_one_epoch(model, loader, optimizer, device="cpu")
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_train_one_epoch_updates_weights(self) -> None:
        """Model parameters must change after one training epoch."""
        set_seed(0)
        model = _make_model()
        loader = _make_synthetic_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Snapshot weights before training
        before = [p.clone().detach() for p in model.parameters()]

        train_one_epoch(model, loader, optimizer, device="cpu")

        # At least one parameter must have changed
        after = list(model.parameters())
        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        assert changed, "Weights must change after train_one_epoch"


# ---------------------------------------------------------------------------
# evaluate_epoch tests
# ---------------------------------------------------------------------------

class TestEvaluateEpoch:
    def test_evaluate_epoch_returns_float(self) -> None:
        """evaluate_epoch must return a Python float."""
        model = _make_model()
        loader = _make_synthetic_dataloader()
        result = evaluate_epoch(model, loader, device="cpu")
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_evaluate_epoch_does_not_update_weights(self) -> None:
        """Model parameters must NOT change after evaluate_epoch."""
        set_seed(0)
        model = _make_model()
        loader = _make_synthetic_dataloader()

        before = [p.clone().detach() for p in model.parameters()]
        evaluate_epoch(model, loader, device="cpu")
        after = list(model.parameters())

        for b, a in zip(before, after):
            assert torch.allclose(b, a), "evaluate_epoch must not modify weights"

    def test_evaluate_epoch_no_grad(self) -> None:
        """No gradient tensors should exist after evaluate_epoch completes."""
        model = _make_model()
        loader = _make_synthetic_dataloader()
        evaluate_epoch(model, loader, device="cpu")

        for p in model.parameters():
            assert p.grad is None or not p.requires_grad or p.grad.sum() == 0, (
                "evaluate_epoch must not accumulate gradients"
            )


# ---------------------------------------------------------------------------
# EarlyStopping tests
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_early_stopping_does_not_fire_on_improvement(self) -> None:
        """EarlyStopping must return False when loss keeps improving."""
        model = _make_model()
        es = EarlyStopping(patience=3)
        for loss in [1.0, 0.8, 0.6, 0.4]:
            should_stop = es.step(loss, model)
            assert not should_stop, f"Should not stop when loss is improving (loss={loss})"

    def test_early_stopping_fires_after_patience_exceeded(self) -> None:
        """EarlyStopping must return True after `patience` non-improving epochs."""
        model = _make_model()
        es = EarlyStopping(patience=3)
        # One good step, then 4 plateauing steps
        es.step(1.0, model)  # improvement
        for _ in range(3):
            result = es.step(1.0, model)
            assert not result or _ == 2, "Should not fire until patience is exhausted"
        # 4th non-improving step → should fire
        should_stop = es.step(1.0, model)
        assert should_stop, "EarlyStopping must fire after patience non-improving epochs"

    def test_early_stopping_stores_best_state_dict_on_improvement(self) -> None:
        """best_state_dict must be non-None after the first improving step."""
        model = _make_model()
        es = EarlyStopping(patience=3)
        es.step(1.0, model)
        assert es.best_state_dict is not None, "best_state_dict must be set after first improvement"

    def test_early_stopping_best_state_dict_is_deep_copy(self) -> None:
        """Modifying model weights after a step must NOT affect best_state_dict."""
        set_seed(0)
        model = _make_model()
        es = EarlyStopping(patience=3)
        es.step(0.5, model)  # stores best

        # Snapshot the stored best for one parameter
        stored_param_name = next(iter(es.best_state_dict))
        stored_tensor = es.best_state_dict[stored_param_name].clone()

        # Mutate the model weights in-place
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        # The stored best must be unchanged
        assert torch.allclose(es.best_state_dict[stored_param_name], stored_tensor), (
            "best_state_dict must be a deep copy — modifying model must not affect it"
        )
