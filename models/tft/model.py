"""
Step 7.3 — TFT model constants and factory function.

The TemporalFusionTransformer is provided by pytorch-forecasting; this module
defines the hyperparameters and feature lists used consistently across train.py,
evaluate.py, and attention_viz.py.
"""
from __future__ import annotations

HORIZONS = [3, 12, 24, 72]
HORIZON_INDICES = [h - 1 for h in HORIZONS]   # 0-based indices into decoder output

MAX_ENCODER_LENGTH    = 168   # 7-day lookback (vs LSTM's 24hr)
MAX_PREDICTION_LENGTH = 72    # full decoder window; HORIZONS are extracted from it

QUANTILES = [0.05, 0.5, 0.95]   # p5/p50/p95 → 90% PI; p50 is the point forecast

# Features known at future timesteps (calendar — always knowable in advance)
KNOWN_REALS: list[str] = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
]

# Features unknown beyond the current timestep
UNKNOWN_REALS: list[str] = [
    "no2", "o3", "pm10", "co",
    "pm25_roll3", "pm25_roll6", "pm25_roll24",
    "pm25_lag1", "pm25_lag3", "pm25_lag24",
    "spatial_pm25_lag1", "spatial_pm25_lag3", "spatial_pm25_roll6",
    "spatial_no2_lag1", "spatial_o3_lag1", "spatial_elev_diff",
]

TARGET      = "pm25"
STATIC_CATS = ["station_id"]

# TFT architecture hyperparameters
HIDDEN_SIZE            = 64
LSTM_LAYERS            = 2
ATTENTION_HEAD_SIZE    = 4
HIDDEN_CONTINUOUS_SIZE = 32
DROPOUT                = 0.1   # lower than LSTM; variable selection acts as regularisation


def build_tft(training_dataset):
    """
    Instantiate a TemporalFusionTransformer from a fitted TimeSeriesDataSet.

    The dataset carries the normaliser and categorical encoder — the same
    dataset (or one built via from_dataset) must be used at inference time.
    """
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        attention_head_size=ATTENTION_HEAD_SIZE,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        dropout=DROPOUT,
        output_size=len(QUANTILES),
        loss=QuantileLoss(QUANTILES),
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=3,
    )
