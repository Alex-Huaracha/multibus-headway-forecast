"""src.models — model package for Fase 5 and Fase 6a.

Public re-exports:
    HeadwayLSTM: flat LSTM encoder for multi-bus headway forecasting (Fase 5).
    SpatialConvLSTM: 1D spatial conv + LSTM encoder (Fase 6a).
    masked_mse_loss: MSE loss gated on a boolean validity mask (shared).
"""
from .lstm import HeadwayLSTM, masked_mse_loss
from .spatial_conv_lstm import SpatialConvLSTM

__all__ = ["HeadwayLSTM", "SpatialConvLSTM", "masked_mse_loss"]
