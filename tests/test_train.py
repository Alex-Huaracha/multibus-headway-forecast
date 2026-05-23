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


# ---------------------------------------------------------------------------
# Wave 3: train_model, grid_search, checkpoint save/load, denormalize
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_train_model_returns_train_result(self) -> None:
        """train_model must return a TrainResult with best_val_loss and state_dict set."""
        from src.train import TrainResult, train_model

        set_seed(42)
        max_N, ctx_dim = 3, 5
        config = TrainConfig(hidden_size=8, num_layers=1, dropout=0.0, lr=1e-3, max_epochs=3, patience=10)
        model = _make_model(max_N=max_N, context_dim=ctx_dim, hidden_size=8)
        train_dl = _make_synthetic_dataloader(max_N=max_N, context_dim=ctx_dim, n_batches=2)
        val_dl = _make_synthetic_dataloader(max_N=max_N, context_dim=ctx_dim, n_batches=1)

        result = train_model(model, train_dl, val_dl, config, device="cpu")

        assert isinstance(result, TrainResult), f"Expected TrainResult, got {type(result)}"
        assert isinstance(result.best_val_loss, float), "best_val_loss must be float"
        assert result.state_dict is not None and len(result.state_dict) > 0, "state_dict must be populated"
        assert result.config is config, "result.config must be the TrainConfig passed in"

    def test_train_model_respects_early_stopping(self) -> None:
        """train_model with patience=2 on non-improving data must stop before max_epochs."""
        from src.train import train_model

        set_seed(99)
        max_N, ctx_dim = 2, 5
        # Use max_epochs=20 and patience=2; non-improving val loss should stop it early
        config = TrainConfig(hidden_size=4, num_layers=1, dropout=0.0, lr=0.0, max_epochs=20, patience=2)
        model = _make_model(max_N=max_N, context_dim=ctx_dim, hidden_size=4)
        # lr=0.0 means weights never change → val loss never improves → stops after patience+1 epochs
        train_dl = _make_synthetic_dataloader(max_N=max_N, context_dim=ctx_dim, n_batches=2)
        val_dl = _make_synthetic_dataloader(max_N=max_N, context_dim=ctx_dim, n_batches=1)

        result = train_model(model, train_dl, val_dl, config, device="cpu")

        assert result.epochs_trained < config.max_epochs, (
            f"Expected early stop before {config.max_epochs} epochs, "
            f"but trained {result.epochs_trained} epochs"
        )


class TestGridSearch:
    def test_grid_constant_has_24_configs(self) -> None:
        """GRID must contain exactly 24 TrainConfig entries (3×2×2×2 combinations)."""
        from src.train import GRID

        assert len(GRID) == 24, f"GRID must have 24 entries, got {len(GRID)}"
        assert all(isinstance(c, TrainConfig) for c in GRID), "All GRID items must be TrainConfig"

    def test_grid_search_returns_results_list(self) -> None:
        """grid_search must return a list of TrainResult with one entry per config, sorted by best_val_loss."""
        from src.train import TrainResult, grid_search

        set_seed(0)
        max_N, ctx_dim = 3, 5
        two_configs = [
            TrainConfig(hidden_size=4, num_layers=1, dropout=0.0, lr=1e-3, max_epochs=2, patience=5),
            TrainConfig(hidden_size=8, num_layers=1, dropout=0.0, lr=1e-3, max_epochs=2, patience=5),
        ]
        train_dl = _make_synthetic_dataloader(max_N=max_N, context_dim=ctx_dim, n_batches=2)
        val_dl = _make_synthetic_dataloader(max_N=max_N, context_dim=ctx_dim, n_batches=1)

        results = grid_search(train_dl, val_dl, max_N=max_N, configs=two_configs, device="cpu")

        assert isinstance(results, list), "grid_search must return a list"
        assert len(results) == 2, f"Expected 2 results (one per config), got {len(results)}"
        assert all(isinstance(r, TrainResult) for r in results), "All items must be TrainResult"
        # Results must be sorted ascending by best_val_loss
        assert results[0].best_val_loss <= results[1].best_val_loss, (
            "grid_search results must be sorted ascending by best_val_loss"
        )


class TestCheckpoint:
    def test_save_checkpoint_creates_files(self, tmp_path) -> None:
        """save_checkpoint must create model.pt and config.json in the given directory."""
        from src.train import TrainResult, save_checkpoint

        config = TrainConfig(hidden_size=8, num_layers=1, dropout=0.0, lr=1e-3)
        model = _make_model(max_N=3, context_dim=5, hidden_size=8)
        result = TrainResult(
            best_val_loss=0.42,
            best_epoch=1,
            state_dict=copy.deepcopy(model.state_dict()),
            config=config,
        )

        ckpt_dir = tmp_path / "checkpoint"
        save_checkpoint(result, ckpt_dir)

        assert (ckpt_dir / "model.pt").exists(), "model.pt must be created by save_checkpoint"
        assert (ckpt_dir / "config.json").exists(), "config.json must be created by save_checkpoint"

    def test_load_checkpoint_restores_model(self, tmp_path) -> None:
        """load_checkpoint must restore a model that produces identical predictions to the saved one."""
        from src.train import TrainResult, load_checkpoint, save_checkpoint

        set_seed(7)
        max_N, ctx_dim = 3, 5
        config = TrainConfig(hidden_size=8, num_layers=1, dropout=0.0, lr=1e-3)
        original_model = _make_model(max_N=max_N, context_dim=ctx_dim, hidden_size=8)
        result = TrainResult(
            best_val_loss=0.1,
            best_epoch=0,
            state_dict=copy.deepcopy(original_model.state_dict()),
            config=config,
        )

        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(result, ckpt_dir)
        restored_model, _ = load_checkpoint(ckpt_dir, max_N=max_N)

        # Forward pass must produce identical predictions
        set_seed(7)
        x = torch.randn(2, 6, max_N + ctx_dim)
        original_model.eval()
        restored_model.eval()
        with torch.no_grad():
            out_original = original_model(x)
            out_restored = restored_model(x)

        assert torch.allclose(out_original, out_restored, atol=1e-6), (
            "Restored model must produce identical predictions to the original"
        )


class TestDenormalize:
    def test_denormalize_predictions_known_value(self) -> None:
        """denormalize_predictions(z=1.0, mean=5.0, std=2.0) must return 7.0 within 1e-6."""
        from src.train import denormalize_predictions

        pred = torch.tensor([1.0])
        result = denormalize_predictions(pred, mean=5.0, std=2.0)

        expected = torch.tensor([7.0])
        assert torch.allclose(result, expected, atol=1e-6), (
            f"Expected 7.0, got {result.item():.8f}"
        )
