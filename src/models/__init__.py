"""src.models — LSTM model package for Fase 5.

Public re-exports:
    HeadwayLSTM: flat LSTM encoder for multi-bus headway forecasting.
    masked_mse_loss: MSE loss gated on a boolean validity mask.
"""
from .lstm import HeadwayLSTM, masked_mse_loss

__all__ = ["HeadwayLSTM", "masked_mse_loss"]
