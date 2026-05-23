"""Training primitives for the LSTM headway-forecasting baseline.

Architecture decisions applied (from design doc):
  AD-1: Caller concatenates input + context before model forward.
        train_one_epoch / evaluate_epoch perform: x = cat([batch["input"], batch["context"]], dim=-1).
  AD-6: Denormalization — pred_minutes = pred_z * (std + 1e-8) + mean.
  AD-7: Reproducibility — torch + cuda + numpy manual seeds.
  AD-9: Early stopping — monitor val masked MSE, in-memory best-state copy.
  INV-NO-COMPILE: torch.compile() is NOT used (Kaggle CUDA compatibility).
  INV-NO-TB: No TensorBoard, MLflow, or Optuna imports.

ACs covered (Wave 2):
  AC-TRAIN-1: train_one_epoch returns float.
  AC-TRAIN-2: Weights change after train_one_epoch.
  AC-TRAIN-3: evaluate_epoch returns float.
  AC-TRAIN-4: No gradients accumulated during evaluate_epoch.
  AC-TRAIN-5: EarlyStopping fires after patience non-improving epochs.
  AC-TRAIN-6: EarlyStopping.best_state_dict is a deep copy of model state.

ACs covered (Wave 3):
  AC-TRAIN-7: save_checkpoint / load_checkpoint round-trip produces identical predictions.
  AC-TRAIN-8: train_model honours early stopping; restores best weights on return.
  AC-GRID-1..5: GRID has 24 entries; grid_search returns sorted list of TrainResult.
  AC-EVAL-1: denormalize_predictions converts z-scored output back to minutes.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.lstm import HeadwayLSTM, masked_mse_loss


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Hyperparameters for one LSTM training run.

    Required:
        hidden_size: Number of LSTM hidden units.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout probability applied between LSTM layers.
        lr:          Adam learning rate.

    Optional (with defaults):
        batch_size:  DataLoader batch size (default 32).
        max_epochs:  Hard ceiling on training epochs (default 50).
        patience:    Early-stopping patience in epochs (default 10).
        seed:        Global random seed (default 42).
    """

    hidden_size: int
    num_layers: int
    dropout: float
    lr: float
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10
    seed: int = 42


@dataclass
class TrainResult:
    """Result of one full training run (returned by train_model / grid_search).

    Attributes
    ----------
    best_val_loss:
        Lowest validation masked MSE achieved during training.
    best_epoch:
        Zero-indexed epoch at which best_val_loss was recorded.
    epochs_trained:
        Total number of epochs actually executed (including the stopping epoch).
    train_losses:
        Chronological list of per-epoch training losses.
    val_losses:
        Chronological list of per-epoch validation losses.
    state_dict:
        Model weights (deep copy) from the best epoch (AD-9).
    config:
        The TrainConfig that produced this result.
    """

    best_val_loss: float
    best_epoch: int
    epochs_trained: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    state_dict: dict[str, Any] = field(default_factory=dict)
    config: TrainConfig = field(default_factory=lambda: TrainConfig(64, 1, 0.0, 1e-3))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility (AD-7).

    Seeds:
    - torch (CPU)
    - torch.cuda (all GPU devices)
    - numpy

    Note: full determinism across CUDA hardware is NOT guaranteed — this seeds
    the major sources of randomness for same-machine reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Per-epoch training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: Any,  # DataLoader or list of dict batches
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> float:
    """Train for one epoch.

    Iterates the loader; for each batch:
    1. Concatenate batch["input"] and batch["context"] along dim=-1 (AD-1).
    2. Forward pass through model → pred: (B, max_N).
    3. Squeeze target and mask from (B, 1, max_N) to (B, max_N).
    4. Compute masked_mse_loss(pred, target, mask).
    5. Backpropagate and step optimizer.

    Parameters
    ----------
    model:
        HeadwayLSTM (or any nn.Module) to train.
    loader:
        Iterable yielding dicts with keys: input, context, target, target_mask.
    optimizer:
        Optimizer (e.g., Adam).
    device:
        Device string or torch.device. Tensors are moved to this device.

    Returns
    -------
    Mean masked MSE loss across all batches (Python float).
    """
    model.train()
    device = torch.device(device)

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        inp = batch["input"].to(device)       # (B, T_in, max_N)
        ctx = batch["context"].to(device)     # (B, T_in, 5)
        target = batch["target"].to(device)   # (B, 1, max_N) or (B, T_out, max_N)
        mask = batch["target_mask"].to(device)  # (B, 1, max_N)

        # AD-1: concatenate headway + context before model forward.
        x = torch.cat([inp, ctx], dim=-1)  # (B, T_in, max_N + 5)

        # Forward pass.
        pred = model(x)  # (B, max_N)

        # Squeeze time dimension from target/mask: (B, 1, max_N) → (B, max_N).
        target_sq = target.squeeze(1)  # (B, max_N)
        mask_sq = mask.squeeze(1)      # (B, max_N) bool

        loss = masked_mse_loss(pred, target_sq, mask_sq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_epoch(
    model: nn.Module,
    loader: Any,  # DataLoader or list of dict batches
    device: str | torch.device = "cpu",
) -> float:
    """Evaluate on val or test set for one epoch.

    Runs in eval mode with no gradient tracking (AC-TRAIN-3, AC-TRAIN-4).

    Parameters
    ----------
    model:
        HeadwayLSTM (or any nn.Module) to evaluate.
    loader:
        Iterable yielding dicts with keys: input, context, target, target_mask.
    device:
        Device string or torch.device.

    Returns
    -------
    Mean masked MSE loss across all batches (Python float).
    """
    model.eval()
    device = torch.device(device)

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device)
            ctx = batch["context"].to(device)
            target = batch["target"].to(device)
            mask = batch["target_mask"].to(device)

            # AD-1: concatenate headway + context before model forward.
            x = torch.cat([inp, ctx], dim=-1)

            pred = model(x)

            # Squeeze time dimension: (B, 1, max_N) → (B, max_N).
            target_sq = target.squeeze(1)
            mask_sq = mask.squeeze(1)

            loss = masked_mse_loss(pred, target_sq, mask_sq)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor validation loss and stop training when no improvement (AD-9).

    Usage
    -----
    es = EarlyStopping(patience=10)
    for epoch in range(max_epochs):
        val_loss = evaluate_epoch(...)
        if es.step(val_loss, model):
            break
    # Restore best weights:
    model.load_state_dict(es.best_state_dict)

    Notes
    -----
    - Improvement is strictly less than the current best.
    - best_state_dict is a deep copy (AD-9, AC-TRAIN-6).
    - Counter resets on improvement; fires when counter > patience.
    """

    def __init__(self, patience: int) -> None:
        self._patience = patience
        self._counter = 0
        self._best_loss: float = float("inf")
        self._best_state_dict: dict[str, Any] | None = None
        self._should_stop: bool = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Process one validation epoch.

        Parameters
        ----------
        val_loss:
            Validation loss for this epoch.
        model:
            The model being trained; state_dict is deep-copied on improvement.

        Returns
        -------
        True if training should stop (patience exceeded), False otherwise.
        """
        if val_loss < self._best_loss:
            # Improvement: reset counter, save best state.
            self._best_loss = val_loss
            self._counter = 0
            self._best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self._should_stop = True

        return self._should_stop

    @property
    def should_stop(self) -> bool:
        """True if patience has been exceeded."""
        return self._should_stop

    @property
    def best_val_loss(self) -> float:
        """Lowest validation loss recorded so far (float('inf') if never improved)."""
        return self._best_loss

    @property
    def best_state_dict(self) -> dict[str, Any] | None:
        """Deep copy of the model state_dict from the best epoch.

        None if step() has never been called with an improving loss.
        """
        return self._best_state_dict


# ---------------------------------------------------------------------------
# Wave 3: GRID constant, train_model, grid_search, checkpoint I/O, denormalize
# ---------------------------------------------------------------------------

# Context vector dimensionality: hour_sin, hour_cos, dow_sin, dow_cos, atypical.
CONTEXT_DIM: int = 5

# Cartesian product: hidden ∈ {32,64,128} × layers ∈ {1,2} × dropout ∈ {0.0,0.2}
# × lr ∈ {1e-3,5e-4} = 24 configurations (AC-GRID-5).
GRID: list[TrainConfig] = [
    TrainConfig(hidden_size=h, num_layers=n, dropout=d, lr=lr)
    for h in [32, 64, 128]
    for n in [1, 2]
    for d in [0.0, 0.2]
    for lr in [1e-3, 5e-4]
]


def train_model(
    model: nn.Module,
    train_dl: Any,
    val_dl: Any,
    config: TrainConfig,
    device: str | torch.device = "cpu",
) -> TrainResult:
    """Full training loop with early stopping (AC-TRAIN-7, AC-TRAIN-8).

    Steps per epoch:
    1. train_one_epoch → training loss.
    2. evaluate_epoch → validation loss.
    3. EarlyStopping.step: save best weights; break if patience exceeded.

    After training: restores best weights into model via load_state_dict.

    Parameters
    ----------
    model:
        HeadwayLSTM (or any nn.Module) to train.
    train_dl:
        Training dataloader (iterable of dicts).
    val_dl:
        Validation dataloader (iterable of dicts).
    config:
        TrainConfig controlling hyperparameters and training policy.
    device:
        Device string or torch.device.

    Returns
    -------
    TrainResult with best_val_loss, best_epoch, epochs_trained, and state_dict.
    """
    set_seed(config.seed)
    model.to(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    early_stopping = EarlyStopping(patience=config.patience)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_epoch: int = 0
    epoch: int = 0

    for epoch in range(config.max_epochs):
        t_loss = train_one_epoch(model, train_dl, optimizer, device)
        v_loss = evaluate_epoch(model, val_dl, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        improved = v_loss < early_stopping.best_val_loss
        if improved:
            best_epoch = epoch

        if early_stopping.step(v_loss, model):
            break

    # Restore best weights (AD-9).
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    return TrainResult(
        best_val_loss=early_stopping.best_val_loss,
        best_epoch=best_epoch,
        epochs_trained=epoch + 1,
        train_losses=train_losses,
        val_losses=val_losses,
        state_dict=early_stopping.best_state_dict or {},
        config=config,
    )


def grid_search(
    train_dl: Any,
    val_dl: Any,
    max_N: int,
    configs: list[TrainConfig],
    device: str | torch.device = "cpu",
) -> list[TrainResult]:
    """Train one model per config; return results sorted ascending by best_val_loss (AC-GRID-1..3).

    Parameters
    ----------
    train_dl:
        Training dataloader.
    val_dl:
        Validation dataloader.
    max_N:
        Maximum number of buses (determines model input_size = max_N + CONTEXT_DIM).
    configs:
        List of TrainConfig to evaluate. Use GRID for the full 24-config sweep.
    device:
        Device string or torch.device.

    Returns
    -------
    List of TrainResult, sorted ascending by best_val_loss (best config first).
    """
    results: list[TrainResult] = []

    for config in configs:
        model = HeadwayLSTM(
            input_size=max_N + CONTEXT_DIM,
            hidden_size=config.hidden_size,
            output_size=max_N,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        result = train_model(model, train_dl, val_dl, config, device)
        results.append(result)

    return sorted(results, key=lambda r: r.best_val_loss)


def save_checkpoint(result: TrainResult, path: Path) -> None:
    """Persist a TrainResult to disk as model.pt + config.json (AC-TRAIN-7).

    Parameters
    ----------
    result:
        TrainResult containing state_dict and config.
    path:
        Directory to create and write files into. Created if absent.

    Files written
    -------------
    model.pt     — torch state_dict (weights_only compatible).
    config.json  — serialised TrainConfig fields plus best_val_loss and epochs_trained.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(result.state_dict, path / "model.pt")

    config_dict: dict[str, Any] = {
        "hidden_size": result.config.hidden_size,
        "num_layers": result.config.num_layers,
        "dropout": result.config.dropout,
        "lr": result.config.lr,
        "batch_size": result.config.batch_size,
        "max_epochs": result.config.max_epochs,
        "patience": result.config.patience,
        "seed": result.config.seed,
        "best_val_loss": result.best_val_loss,
        "epochs_trained": result.epochs_trained,
    }
    (path / "config.json").write_text(json.dumps(config_dict, indent=2))


def load_checkpoint(path: Path, max_N: int) -> tuple[HeadwayLSTM, TrainConfig]:
    """Restore a HeadwayLSTM and its TrainConfig from a checkpoint directory (AC-TRAIN-7).

    Parameters
    ----------
    path:
        Directory previously written by save_checkpoint.
    max_N:
        Maximum number of buses; used to reconstruct model input_size and output_size.

    Returns
    -------
    (model, config) where model weights match the saved state_dict.
    """
    path = Path(path)
    config_dict = json.loads((path / "config.json").read_text())

    config = TrainConfig(
        hidden_size=config_dict["hidden_size"],
        num_layers=config_dict["num_layers"],
        dropout=config_dict["dropout"],
        lr=config_dict["lr"],
        batch_size=config_dict.get("batch_size", 32),
        max_epochs=config_dict.get("max_epochs", 50),
        patience=config_dict.get("patience", 10),
        seed=config_dict.get("seed", 42),
    )

    model = HeadwayLSTM(
        input_size=max_N + CONTEXT_DIM,
        hidden_size=config.hidden_size,
        output_size=max_N,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    state_dict = torch.load(path / "model.pt", weights_only=True)
    model.load_state_dict(state_dict)
    return model, config


def denormalize_predictions(
    pred: torch.Tensor,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Convert z-scored predictions back to original scale (minutes) (AD-6, AC-EVAL-1).

    Formula: pred_minutes = pred_z * (std + 1e-8) + mean

    Parameters
    ----------
    pred:
        Tensor of z-scored predictions from the model.
    mean:
        Per-corridor mean (in minutes) used during z-scoring.
    std:
        Per-corridor standard deviation (in minutes) used during z-scoring.

    Returns
    -------
    Tensor of predicted headways in minutes (same shape as pred).
    """
    return pred * (std + 1e-8) + mean
