"""
Step 8.2 — DeepAR constants and estimator factory.

DeepAR via GluonTS (PyTorch backend). Autoregressive RNN outputting full
predictive distributions via Monte Carlo sampling. ISQF output chosen to
handle the asymmetric upper tail of PM2.5 during wildfire/inversion events.

Key design choices vs TFT:
  - Context length matches TFT encoder: 168hr (7-day) lookback.
  - Prediction length matches TFT decoder: 72hr.
  - Only calendar features (4) passed as feat_dynamic_real — the only
    covariates genuinely known in the future in production.
  - PM2.5 lags handled via lags_seq=[1,3,24]: GluonTS computes these
    internally and uses model predictions (not actual future values)
    during inference, avoiding target leakage.
  - All other covariates (NO2, O3, pollutant lags, spatial features)
    dropped — not available in production at forecast time.
  - 1 static categorical: station_id (GluonTS embeds it automatically).
  - ISQFOutput (10 pieces, 9 interior knots): directly learns the quantile
    function without parametric assumptions. Superior to StudentT for
    asymmetric tails — StudentT v2 showed 20% of actuals above p95 at h3/h12
    because the distribution was too symmetric to capture wildfire spikes.
  - 500 Monte Carlo samples: enough for stable breach-probability estimates
    at the 5% tail (±1% MC error at N=500).
  - Horizon-weighted CRPS (v6): exponential decay w(t)=exp(-t/τ), τ=24h.
    h3 gets ~1.5x weight vs h24, h72 gets ~0.08x. Up-weights the near-term
    horizons (h3/h12) that showed systematic undercoverage in v4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from gluonts.torch.distributions import ISQFOutput
from gluonts.torch.distributions.isqf import TransformedISQF

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

# Only calendar features are genuinely known in the future in production.
# PM2.5 lags are handled internally by lags_seq (no leakage).
# Pollutant covariates, spatial features, and PM2.5 rolls are not available
# at forecast time and are excluded.
DYNAMIC_REAL_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
]

# PM2.5 autoregressive lags passed to GluonTS internally.
# During inference GluonTS feeds the model's own predictions back as lags
# rather than actual observed values, so there is no target leakage.
LAGS_SEQ = [1, 3, 24]

NUM_FEAT_DYNAMIC_REAL = len(DYNAMIC_REAL_FEATURES)   # 4
NUM_FEAT_STATIC_CAT   = 1                             # station_id

TARGET    = "pm25"
QUANTILES = [0.05, 0.5, 0.95]   # for summary statistics; samples used for alerts

# ISQF hyperparameters (v4/v6 — active predictor)
# num_pieces=10: piecewise-linear spline with 10 segments.
# qk_x: 0.05 and 0.95 as explicit endpoint knots so the PI boundaries are
#   directly learned by the spline rather than extrapolated by the exponential
#   tail model. v5 trial with [0.025..0.975] overfit — intervals collapsed
#   from 13.9 to 7.1 ug/m3 and coverage dropped to 47%.
ISQF_NUM_PIECES = 10
ISQF_QK_X = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# Horizon-weighted CRPS: exponential decay w(t) = exp(-t / τ), τ in steps.
# At τ=24: w(h3)≈0.88, w(h12)≈0.63, w(h24)≈0.38, w(h72)≈0.05 (pre-norm).
# Normalized so w.mean()=1, keeping the overall CRPS magnitude comparable
# across runs with and without weighting.
HORIZON_WEIGHT_TAU = 24.0

# ---------------------------------------------------------------------------
# ISQF compatibility shim
# ---------------------------------------------------------------------------

class FixedISQFOutput(ISQFOutput):
    """
    ISQFOutput subclass fixing two GluonTS 0.16.2 + PyTorch ≥2.x incompatibilities
    and adding horizon-weighted CRPS training:

    1. loc=None crash: DeepAR calls distr_output.loss(scale=scale) without loc,
       which passes loc=None into AffineTransform. PyTorch ≥2.x treats None as
       a missing operand in AffineTransform._inverse → TypeError. Fix: replace
       None with zeros_like(scale) before constructing the transform.

    2. Wrong loss: ISQFOutput doesn't override DistributionOutput.loss(), so it
       falls back to -log_prob(). ISQF is a quantile-function model, not a
       density model, so CRPS is the correct training objective. Fix: override
       loss() to call distribution.crps(), which TransformedISQF implements
       analytically with correct affine rescaling.

    3. Horizon weighting (v6): per-step exponential decay w(t)=exp(-t/τ),
       normalized so mean(w)=1. Up-weights h3/h12 to address their systematic
       undercoverage vs the far horizon. τ=HORIZON_WEIGHT_TAU (default 24 steps).
    """

    def distribution(
        self,
        distr_args: Tuple,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> TransformedISQF:
        if loc is None and scale is not None:
            loc = torch.zeros_like(scale)
        return super().distribution(distr_args, loc=loc, scale=scale)

    def loss(
        self,
        target: torch.Tensor,
        distr_args: Tuple,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if loc is None and scale is not None:
            loc = torch.zeros_like(scale)
        distr = self.distribution(distr_args, loc=loc, scale=scale)
        crps = distr.crps(target)   # (batch, T)
        if crps.dim() >= 2:
            T = crps.shape[-1]
            t = torch.arange(T, device=crps.device, dtype=crps.dtype)
            w = torch.exp(-t / HORIZON_WEIGHT_TAU)
            w = w / w.mean()  # normalize so overall loss scale is comparable to unweighted
            crps = crps * w
        return crps


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------

MODEL_DIR      = Path(__file__).parent
PREDICTOR_PATH = MODEL_DIR / "predictor"


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

    default_trainer_kwargs = {
        "max_epochs":          50,
        "accelerator":         "auto",
        "gradient_clip_val":   0.1,
        # ModelSummary runs a forward pass during fit() setup, which calls
        # ISQFOutput.sample() before distribution params are set — loc=None crash.
        "enable_model_summary": False,
    }
    if trainer_kwargs is not None:
        default_trainer_kwargs.update(trainer_kwargs)

    return DeepAREstimator(
        freq=FREQ,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        distr_output=FixedISQFOutput(num_pieces=ISQF_NUM_PIECES, qk_x=ISQF_QK_X),
        num_feat_dynamic_real=NUM_FEAT_DYNAMIC_REAL,
        num_feat_static_cat=NUM_FEAT_STATIC_CAT,
        cardinality=cardinality,
        lags_seq=LAGS_SEQ,
        num_batches_per_epoch=100,
        trainer_kwargs=default_trainer_kwargs,
    )
