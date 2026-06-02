"""src.models — model package for Fase 5, Fase 6a, and Fase 6b.

Public re-exports:
    HeadwayLSTM: flat LSTM encoder for multi-bus headway forecasting (Fase 5).
    SpatialConvLSTM: 1D spatial conv + LSTM encoder (Fase 6a).
    SpatialTransformer: MHA spatial encoder + LSTM temporal encoder (Fase 6b).
    masked_mse_loss: MSE loss gated on a boolean validity mask (shared).
"""
from .lstm import HeadwayLSTM, masked_mse_loss
from .spatial_conv_lstm import SpatialConvLSTM
from .spatial_transformer import SpatialTransformer

__all__ = ["HeadwayLSTM", "SpatialConvLSTM", "SpatialTransformer", "masked_mse_loss"]
