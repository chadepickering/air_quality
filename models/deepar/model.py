"""
Step 8.2 — DeepAR constants and estimator factory.

DeepAR via GluonTS (PyTorch backend). Autoregressive RNN outputting full
predictive distributions via Monte Carlo sampling. StudentT output chosen
for heavy-tailed PM2.5 behavior during wildfire and inversion events.

Key design choices vs TFT:
  - Context length matches TFT encoder: 168hr (7-day) lookback.
  - Prediction length matches TFT decoder: 72hr.
  - All 20 dynamic real features passed as past covariates.
  - 1 static categorical: station_id (GluonTS embeds it automatically).
  - StudentT output: heavier tails than Gaussian — better coverage during
    extreme smoke events without over-widening intervals in clean periods.
  - 500 Monte Carlo samples: enough for stable breach-probability estimates
    at the 5% tail (±1% MC error at N=500).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Horizon configuration (consistent with LSTM and TFT)
# ---------------------------------------------------------------------------

HORIZONS        = [3, 12, 24, 72]          # forecast hours of interest
HORIZON_INDICES = [h - 1 for h in HORIZONS]  # 0-based indices into 72-step output

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

FREQ              = "h"     # hourly data
CONTEXT_LENGTH    = 168     # 7-day encoder lookback
PREDICTION_LENGTH = 72      # 3-day forecast horizon

NUM_SAMPLES = 500           # Monte Carlo trajectories per forecast window

# Dynamic real features: all 20 time-varying columns (calendar + pollutant/lag/spatial)
DYNAMIC_REAL_FEATURES = [
    # Calendar (knowable in advance)
    "hour_of_day", "day_of_week", "month", "is_weekend",
    # Pollutant covariates
    "no2", "o3", "pm10", "co",
    # PM2.5 rolling and lag
    "pm25_roll3", "pm25_roll6", "pm25_roll24",
    "pm25_lag1", "pm25_lag3", "pm25_lag24",
    # Spatial lag features
    "spatial_pm25_lag1", "spatial_pm25_lag3", "spatial_pm25_roll6",
    "spatial_no2_lag1", "spatial_o3_lag1",
    # Spatial elevation differential (static in practice, included as dynamic for simplicity)
    "spatial_elev_diff",
]

NUM_FEAT_DYNAMIC_REAL = len(DYNAMIC_REAL_FEATURES)   # 20
NUM_FEAT_STATIC_CAT   = 1                             # station_id

TARGET    = "pm25"
QUANTILES = [0.05, 0.5, 0.95]   # for summary statistics; samples used for alerts

# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent


def build_estimator(
    cardinality: list[int],
    trainer_kwargs: dict | None = None,
) -> Any:
    """
    Build a DeepAREstimator with project-standard hyperparameters.

    Args:
        cardinality:    list with one entry — number of unique station_ids.
                        GluonTS uses this to size the static embedding.
        trainer_kwargs: overrides the default trainer config (used by train.py
                        to inject W&B logger and EarlyStopping callback).
    """
    from gluonts.torch.model.deepar import DeepAREstimator
    from gluonts.torch.distributions import StudentTOutput

    default_trainer_kwargs = {
        "max_epochs":        50,
        "accelerator":       "auto",
        "gradient_clip_val": 0.1,
    }
    if trainer_kwargs is not None:
        default_trainer_kwargs.update(trainer_kwargs)

    return DeepAREstimator(
        freq=FREQ,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        distr_output=StudentTOutput(),
        num_feat_dynamic_real=NUM_FEAT_DYNAMIC_REAL,
        num_feat_static_cat=NUM_FEAT_STATIC_CAT,
        cardinality=cardinality,
        num_batches_per_epoch=100,
        trainer_kwargs=default_trainer_kwargs,
    )
