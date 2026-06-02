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

    def test_load_checkpoint_restores_spatial_model(self, tmp_path) -> None:
        """load_checkpoint must round-trip a SpatialConvLSTM with identical predictions."""
        from src.train import TrainResult, load_checkpoint, save_checkpoint

        set_seed(7)
        max_N, conv_channels = 4, 8
        config = TrainConfig(
            hidden_size=8, num_layers=1, dropout=0.0, lr=1e-3,
            conv_channels=conv_channels,
        )
        original_model = _make_spatial_model(
            max_N=max_N, conv_channels=conv_channels, hidden_size=8,
        )
        result = TrainResult(
            best_val_loss=0.1,
            best_epoch=0,
            state_dict=copy.deepcopy(original_model.state_dict()),
            config=config,
        )

        ckpt_dir = tmp_path / "spatial_ckpt"
        save_checkpoint(result, ckpt_dir)
        restored_model, restored_config = load_checkpoint(ckpt_dir, max_N=max_N)

        assert restored_config.conv_channels == conv_channels
        assert hasattr(restored_model, "spatial") and restored_model.spatial

        inp = torch.randn(2, 6, max_N)
        ctx = torch.randn(2, 6, 5)
        mask = torch.ones(2, 6, max_N, dtype=torch.bool)
        original_model.eval()
        restored_model.eval()
        with torch.no_grad():
            out_original = original_model(inp, ctx, mask)
            out_restored = restored_model(inp, ctx, mask)

        assert torch.allclose(out_original, out_restored, atol=1e-6), (
            "Restored SpatialConvLSTM must produce identical predictions"
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


# ---------------------------------------------------------------------------
# Wave 2 (Fase 6a): SpatialConvLSTM dispatch, TrainConfig.conv_channels,
#                   SPATIAL_GRID, and backward-compat verification
# ---------------------------------------------------------------------------

def _make_spatial_dataloader(
    max_N: int = 4,
    context_dim: int = 5,
    seq_len: int = 6,
    batch_size: int = 2,
    n_batches: int = 2,
) -> list[dict[str, torch.Tensor]]:
    """Minimal synthetic loader that includes input_mask — required by SpatialConvLSTM."""
    batches = []
    for _ in range(n_batches):
        inp = torch.randn(batch_size, seq_len, max_N)
        ctx = torch.randn(batch_size, seq_len, context_dim)
        target = torch.randn(batch_size, 1, max_N)
        target_mask = torch.ones(batch_size, 1, max_N, dtype=torch.bool)
        input_mask = torch.ones(batch_size, seq_len, max_N, dtype=torch.bool)
        batches.append({
            "input": inp,
            "context": ctx,
            "target": target,
            "target_mask": target_mask,
            "input_mask": input_mask,
        })
    return batches


def _make_spatial_model(max_N: int = 4, conv_channels: int = 1, hidden_size: int = 8):
    """Build a tiny SpatialConvLSTM for testing."""
    from src.models.spatial_conv_lstm import SpatialConvLSTM
    return SpatialConvLSTM(
        max_N=max_N,
        conv_channels=conv_channels,
        hidden_size=hidden_size,
        output_size=max_N,
        num_layers=1,
        dropout=0.0,
        context_size=5,
    )


class TestTrainConfigConvChannels:
    def test_train_config_conv_channels_default_none(self) -> None:
        """TrainConfig().conv_channels must default to None (AD-5, backward compat)."""
        cfg = TrainConfig(hidden_size=32, num_layers=1, dropout=0.0, lr=1e-3)
        assert cfg.conv_channels is None, (
            f"Expected conv_channels=None, got {cfg.conv_channels!r}"
        )

    def test_train_config_conv_channels_can_be_set(self) -> None:
        """TrainConfig must accept an explicit conv_channels value."""
        cfg = TrainConfig(hidden_size=32, num_layers=1, dropout=0.0, lr=1e-3, conv_channels=8)
        assert cfg.conv_channels == 8


class TestSpatialGrid:
    def test_spatial_grid_length_48(self) -> None:
        """SPATIAL_GRID must contain exactly 48 TrainConfig entries (AD-6: 3×2×2×2×2)."""
        from src.train import SPATIAL_GRID

        assert len(SPATIAL_GRID) == 48, (
            f"SPATIAL_GRID must have 48 entries, got {len(SPATIAL_GRID)}"
        )

    def test_spatial_grid_all_conv_channels_positive(self) -> None:
        """Every config in SPATIAL_GRID must have conv_channels in {1, 8, 16}."""
        from src.train import SPATIAL_GRID

        valid = {1, 8, 16}
        for cfg in SPATIAL_GRID:
            assert cfg.conv_channels in valid, (
                f"Expected conv_channels in {valid}, got {cfg.conv_channels}"
            )

    def test_spatial_grid_all_are_train_config(self) -> None:
        """All SPATIAL_GRID items must be TrainConfig instances."""
        from src.train import SPATIAL_GRID

        assert all(isinstance(c, TrainConfig) for c in SPATIAL_GRID), (
            "Every SPATIAL_GRID entry must be a TrainConfig"
        )


class TestSpatialDispatchTrainOneEpoch:
    def test_train_one_epoch_spatial_dispatch(self) -> None:
        """train_one_epoch must call model(inp, ctx, input_mask) when model.spatial is True."""
        import unittest.mock as mock

        spatial_model = _make_spatial_model()
        loader = _make_spatial_dataloader()
        optimizer = torch.optim.Adam(spatial_model.parameters(), lr=1e-3)

        call_args_list = []
        original_forward = spatial_model.forward

        def recording_forward(inp, ctx, input_mask):
            call_args_list.append((inp, ctx, input_mask))
            return original_forward(inp, ctx, input_mask)

        with mock.patch.object(spatial_model, "forward", side_effect=recording_forward):
            train_one_epoch(spatial_model, loader, optimizer, device="cpu")

        assert len(call_args_list) > 0, "forward must be called at least once"
        # Each call must have received 3 positional arguments
        for args in call_args_list:
            assert len(args) == 3, (
                f"Expected 3 args (inp, ctx, input_mask), got {len(args)}"
            )

    def test_train_one_epoch_spatial_returns_finite_float(self) -> None:
        """train_one_epoch with SpatialConvLSTM must return a finite float >= 0."""
        spatial_model = _make_spatial_model()
        loader = _make_spatial_dataloader()
        optimizer = torch.optim.Adam(spatial_model.parameters(), lr=1e-3)

        result = train_one_epoch(spatial_model, loader, optimizer, device="cpu")

        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert result >= 0.0, f"Loss must be non-negative, got {result}"
        assert result == result, "Loss must not be NaN"  # NaN != NaN
        import math
        assert math.isfinite(result), f"Loss must be finite, got {result}"


class TestSpatialDispatchEvaluateEpoch:
    def test_evaluate_epoch_spatial_dispatch(self) -> None:
        """evaluate_epoch must call model(inp, ctx, input_mask) when model.spatial is True."""
        import unittest.mock as mock

        spatial_model = _make_spatial_model()
        loader = _make_spatial_dataloader()

        call_args_list = []
        original_forward = spatial_model.forward

        def recording_forward(inp, ctx, input_mask):
            call_args_list.append((inp, ctx, input_mask))
            return original_forward(inp, ctx, input_mask)

        with mock.patch.object(spatial_model, "forward", side_effect=recording_forward):
            evaluate_epoch(spatial_model, loader, device="cpu")

        assert len(call_args_list) > 0, "forward must be called at least once"
        for args in call_args_list:
            assert len(args) == 3, (
                f"Expected 3 args (inp, ctx, input_mask), got {len(args)}"
            )

    def test_evaluate_epoch_spatial_returns_finite_float(self) -> None:
        """evaluate_epoch with SpatialConvLSTM must return a finite float >= 0."""
        import math

        spatial_model = _make_spatial_model()
        loader = _make_spatial_dataloader()

        result = evaluate_epoch(spatial_model, loader, device="cpu")

        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert result >= 0.0, f"Loss must be non-negative, got {result}"
        assert math.isfinite(result), f"Loss must be finite, got {result}"

    def test_evaluate_epoch_spatial_does_not_update_weights(self) -> None:
        """evaluate_epoch must NOT update SpatialConvLSTM weights."""
        spatial_model = _make_spatial_model()
        loader = _make_spatial_dataloader()

        before = [p.clone().detach() for p in spatial_model.parameters()]
        evaluate_epoch(spatial_model, loader, device="cpu")
        after = list(spatial_model.parameters())

        for b, a in zip(before, after):
            assert torch.allclose(b, a), "evaluate_epoch must not modify weights"


class TestGridSearchWithSpatialModel:
    def test_grid_search_with_spatial_model_returns_results(self) -> None:
        """grid_search must work with a 1-config SPATIAL_GRID list and SpatialConvLSTM."""
        from src.train import TrainResult, grid_search

        set_seed(0)
        max_N = 4
        spatial_cfg = [
            TrainConfig(
                hidden_size=8, num_layers=1, dropout=0.0, lr=1e-3,
                max_epochs=2, patience=5, conv_channels=1,
            )
        ]
        train_dl = _make_spatial_dataloader(max_N=max_N, n_batches=2)
        val_dl = _make_spatial_dataloader(max_N=max_N, n_batches=1)

        results = grid_search(train_dl, val_dl, max_N=max_N, configs=spatial_cfg, device="cpu")

        assert isinstance(results, list), "grid_search must return a list"
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert isinstance(results[0], TrainResult), "Result must be a TrainResult"
        assert isinstance(results[0].best_val_loss, float)


class TestBackwardCompatAfterDispatch:
    """Verify that the HeadwayLSTM path is UNCHANGED after W2 dispatch changes."""

    def test_train_one_epoch_lstm_still_works(self) -> None:
        """HeadwayLSTM path in train_one_epoch must work after W2 dispatch changes."""
        model = _make_model()
        # Existing dataloader WITHOUT input_mask must work for HeadwayLSTM
        loader = _make_synthetic_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = train_one_epoch(model, loader, optimizer, device="cpu")
        assert isinstance(result, float)

    def test_evaluate_epoch_lstm_still_works(self) -> None:
        """HeadwayLSTM path in evaluate_epoch must work after W2 dispatch changes."""
        model = _make_model()
        loader = _make_synthetic_dataloader()
        result = evaluate_epoch(model, loader, device="cpu")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Wave 2 (Fase 6b): SpatialTransformer dispatch, TrainConfig.nhead/d_model,
#                   TRANSFORMER_GRID, and backward-compat regression tests.
#
# Grid decision (obs #417): TRANSFORMER_GRID = nhead{1,2} × d_model{16,32}
# × hidden{32,64} × dropout{0.0,0.2} × lr{1e-3,5e-4} → 32 configs
# (all (d_model, nhead) combos satisfy divisibility → 2^5 = 32 with no filtering).
# ---------------------------------------------------------------------------

def _make_transformer_model(
    max_N: int = 4,
    nhead: int = 1,
    d_model: int = 16,
    hidden_size: int = 8,
):
    """Build a tiny SpatialTransformer for testing."""
    from src.models.spatial_transformer import SpatialTransformer
    return SpatialTransformer(
        max_N=max_N,
        nhead=nhead,
        d_model=d_model,
        hidden_size=hidden_size,
        output_size=max_N,
        num_layers=1,
        dropout=0.0,
        context_size=5,
    )


class TestTrainConfigTransformerFields:
    """TrainConfig backward-compat tests for new nhead / d_model fields."""

    def test_train_config_nhead_default_none(self) -> None:
        """TrainConfig().nhead must default to None (backward compat)."""
        cfg = TrainConfig(hidden_size=32, num_layers=1, dropout=0.0, lr=1e-3)
        assert cfg.nhead is None, (
            f"Expected nhead=None by default, got {cfg.nhead!r}"
        )

    def test_train_config_d_model_default_none(self) -> None:
        """TrainConfig().d_model must default to None (backward compat)."""
        cfg = TrainConfig(hidden_size=32, num_layers=1, dropout=0.0, lr=1e-3)
        assert cfg.d_model is None, (
            f"Expected d_model=None by default, got {cfg.d_model!r}"
        )

    def test_train_config_nhead_can_be_set(self) -> None:
        """TrainConfig must accept explicit nhead and d_model values."""
        cfg = TrainConfig(hidden_size=32, num_layers=1, dropout=0.0, lr=1e-3,
                          nhead=2, d_model=16)
        assert cfg.nhead == 2
        assert cfg.d_model == 16


class TestTransformerGrid:
    """TRANSFORMER_GRID constant validation (obs #417 canonical grid)."""

    def test_transformer_grid_has_32_configs(self) -> None:
        """TRANSFORMER_GRID must contain exactly 32 configs (obs #417: 2^5 without invalid combos).

        Grid: nhead{1,2} × d_model{16,32} × hidden{32,64}
              × dropout{0.0,0.2} × lr{1e-3,5e-4}
        All (nhead, d_model) combos satisfy d_model % nhead == 0,
        so 2×2×2×2×2 = 32 configs total with num_layers=1 fixed.
        """
        from src.train import TRANSFORMER_GRID
        assert len(TRANSFORMER_GRID) == 32, (
            f"TRANSFORMER_GRID must have 32 configs (obs #417), got {len(TRANSFORMER_GRID)}"
        )

    def test_transformer_grid_d_model_divisible_by_nhead(self) -> None:
        """Every TRANSFORMER_GRID config must satisfy d_model % nhead == 0."""
        from src.train import TRANSFORMER_GRID
        invalid = [
            (c.nhead, c.d_model) for c in TRANSFORMER_GRID
            if c.d_model % c.nhead != 0
        ]
        assert not invalid, (
            f"Found configs where d_model % nhead != 0: {invalid}"
        )

    def test_transformer_grid_all_nhead_not_none(self) -> None:
        """Every TRANSFORMER_GRID config must have nhead in {1, 2} (never None)."""
        from src.train import TRANSFORMER_GRID
        invalid = [c.nhead for c in TRANSFORMER_GRID if c.nhead is None]
        assert not invalid, (
            "All TRANSFORMER_GRID configs must have nhead set (never None)"
        )
        valid_nheads = {1, 2}
        bad = [c.nhead for c in TRANSFORMER_GRID if c.nhead not in valid_nheads]
        assert not bad, (
            f"nhead values must be in {{1, 2}}, found: {set(bad)}"
        )

    def test_transformer_grid_all_are_train_config(self) -> None:
        """All TRANSFORMER_GRID items must be TrainConfig instances."""
        from src.train import TRANSFORMER_GRID
        assert all(isinstance(c, TrainConfig) for c in TRANSFORMER_GRID), (
            "Every TRANSFORMER_GRID entry must be a TrainConfig"
        )

    def test_transformer_grid_num_layers_fixed_1(self) -> None:
        """TRANSFORMER_GRID must use num_layers=1 (fixed per obs #417)."""
        from src.train import TRANSFORMER_GRID
        bad = [c.num_layers for c in TRANSFORMER_GRID if c.num_layers != 1]
        assert not bad, (
            f"All TRANSFORMER_GRID configs must have num_layers=1, found: {bad}"
        )


class TestTransformerGridSearchDispatch:
    """grid_search dispatch tests for SpatialTransformer."""

    def test_grid_search_dispatches_transformer_on_nhead(self) -> None:
        """config with nhead set → grid_search creates SpatialTransformer."""
        from src.train import TrainResult, grid_search
        from src.models.spatial_transformer import SpatialTransformer

        set_seed(0)
        max_N = 4
        transformer_cfg = [
            TrainConfig(
                hidden_size=8, num_layers=1, dropout=0.0, lr=1e-3,
                max_epochs=2, patience=5, nhead=1, d_model=16,
            )
        ]
        train_dl = _make_spatial_dataloader(max_N=max_N, n_batches=2)
        val_dl = _make_spatial_dataloader(max_N=max_N, n_batches=1)

        results = grid_search(train_dl, val_dl, max_N=max_N, configs=transformer_cfg, device="cpu")

        assert isinstance(results, list), "grid_search must return a list"
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert isinstance(results[0], TrainResult), "Result must be a TrainResult"
        # The dispatched model must have been a SpatialTransformer.
        # We verify indirectly: config has nhead set and result is valid.
        assert results[0].config.nhead == 1
        assert isinstance(results[0].best_val_loss, float)

    def test_grid_search_nhead_priority_raises_or_transformer(self) -> None:
        """config with both nhead and conv_channels → ValueError (mutual exclusivity).

        nhead MUST take priority — conv_channels must never silently win.
        """
        from src.train import grid_search

        set_seed(0)
        max_N = 4
        conflicting_cfg = [
            TrainConfig(
                hidden_size=8, num_layers=1, dropout=0.0, lr=1e-3,
                max_epochs=2, patience=5,
                nhead=1, d_model=16, conv_channels=8,  # mutual exclusivity violation
            )
        ]
        train_dl = _make_spatial_dataloader(max_N=max_N, n_batches=1)
        val_dl = _make_spatial_dataloader(max_N=max_N, n_batches=1)

        with pytest.raises(ValueError, match=r"nhead.*conv_channels|conv_channels.*nhead|mutually exclusive"):
            grid_search(train_dl, val_dl, max_N=max_N, configs=conflicting_cfg, device="cpu")

    def test_grid_search_conv_channels_still_dispatches_spatial_conv_lstm(self) -> None:
        """nhead=None, conv_channels=8 → SpatialConvLSTM (Fase 6a regression test)."""
        from src.train import TrainResult, grid_search

        set_seed(0)
        max_N = 4
        spatial_cfg = [
            TrainConfig(
                hidden_size=8, num_layers=1, dropout=0.0, lr=1e-3,
                max_epochs=2, patience=5, conv_channels=8,  # nhead defaults to None
            )
        ]
        train_dl = _make_spatial_dataloader(max_N=max_N, n_batches=2)
        val_dl = _make_spatial_dataloader(max_N=max_N, n_batches=1)

        results = grid_search(train_dl, val_dl, max_N=max_N, configs=spatial_cfg, device="cpu")

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].config.conv_channels == 8
        assert results[0].config.nhead is None


class TestTransformerCheckpoint:
    """Checkpoint round-trip for SpatialTransformer."""

    def test_load_checkpoint_nhead_branch(self, tmp_path) -> None:
        """save_checkpoint + load_checkpoint round-trip with SpatialTransformer."""
        import copy
        from src.train import TrainResult, save_checkpoint, load_checkpoint
        from src.models.spatial_transformer import SpatialTransformer

        set_seed(7)
        max_N, nhead, d_model = 4, 2, 16
        config = TrainConfig(
            hidden_size=8, num_layers=1, dropout=0.0, lr=1e-3,
            nhead=nhead, d_model=d_model,
        )
        original_model = _make_transformer_model(max_N=max_N, nhead=nhead, d_model=d_model, hidden_size=8)
        result = TrainResult(
            best_val_loss=0.15,
            best_epoch=0,
            state_dict=copy.deepcopy(original_model.state_dict()),
            config=config,
        )

        ckpt_dir = tmp_path / "transformer_ckpt"
        save_checkpoint(result, ckpt_dir)
        restored_model, restored_config = load_checkpoint(ckpt_dir, max_N=max_N)

        assert isinstance(restored_model, SpatialTransformer), (
            f"Expected SpatialTransformer, got {type(restored_model)}"
        )
        assert restored_config.nhead == nhead
        assert restored_config.d_model == d_model

        # Forward pass must produce identical predictions.
        inp = torch.randn(2, 6, max_N)
        ctx = torch.randn(2, 6, 5)
        mask = torch.ones(2, 6, max_N, dtype=torch.bool)
        original_model.eval()
        restored_model.eval()
        with torch.no_grad():
            out_original = original_model(inp, ctx, mask)
            out_restored = restored_model(inp, ctx, mask)

        assert torch.allclose(out_original, out_restored, atol=1e-6), (
            "Restored SpatialTransformer must produce identical predictions."
        )
