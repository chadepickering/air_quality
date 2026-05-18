"""
Split-conformal prediction calibration for DeepAR PI coverage.

Computes per-horizon asymmetric nonconformity scores on the val set, then
applies the resulting margins to test-set samples to produce coverage-
guaranteed prediction intervals.

Theory (split conformal):
    Given n calibration residuals {s_1, ..., s_n} and target coverage 1-α:
        margin = np.quantile(scores, ceil((n+1)*(1-α)) / n)
    Under exchangeability this guarantees P(y ∈ PI) >= 1-α on new data.

    We use asymmetric nonconformity scores:
        s_upper_i(h) = max(y_i(h) - q95_i(h), 0)   # how far above p95
        s_lower_i(h) = max(q05_i(h) - y_i(h), 0)   # how far below p05
    and compute separate margins for each horizon h, so h12 gets a larger
    adjustment than h24 if it is more miscalibrated.

Usage (from project root, using venv_deepar):
    source venv_deepar/bin/activate
    python -m evaluation.conformal [--alpha 0.10] [--stride-hours 24]

Outputs:
    evaluation/conformal_margins.json  — per-horizon upper/lower margins (μg/m³)
    evaluation/deepar_metrics_conformal.json — test metrics after adjustment
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.deepar.model import (
    CONTEXT_LENGTH,
    FREQ,
    HORIZON_INDICES,
    HORIZONS,
    NUM_SAMPLES,
    PREDICTOR_PATH,
    TARGET,
)
from models.deepar.sample_forecasts import (
    STRIDE_HOURS,
    _crps_energy,
    _make_rolling_instances,
)
from models.deepar.train import load_and_prepare_df

EVAL_DIR               = Path(__file__).parent
MARGINS_OUTPUT         = EVAL_DIR / "conformal_margins.json"
CONFORMAL_METRICS_OUT  = EVAL_DIR / "deepar_metrics_conformal.json"
SAMPLES_PATH           = EVAL_DIR / "deepar_samples.npz"
HORIZON_LABELS         = [f"h{h}" for h in HORIZONS]


# ---------------------------------------------------------------------------
# Val calibration
# ---------------------------------------------------------------------------

def calibrate(
    predictor_path: Path = PREDICTOR_PATH,
    alpha: float = 0.10,
    stride_hours: int = STRIDE_HOURS,
) -> dict[str, dict[str, float]]:
    """
    Run the predictor on rolling val windows and return per-horizon conformal
    margins as {"h3": {"upper": ..., "lower": ...}, ...}.
    """
    from gluonts.dataset.common import ListDataset
    from gluonts.model.predictor import Predictor
    from streaming.feature_engineering import TRAIN_END, VAL_END

    print(f"Loading predictor from {predictor_path}...")
    predictor = Predictor.deserialize(predictor_path)

    # Load train+val so context windows before VAL_START have enough history.
    print("Loading train+val data for conformal calibration...")
    df = load_and_prepare_df(["train", "val"])
    station_ids    = sorted(df["station_id"].unique().tolist())
    station_to_idx = {sid: i for i, sid in enumerate(station_ids)}

    val_start = pd.Timestamp(str(TRAIN_END), tz=None) + pd.Timedelta(hours=1)
    val_end   = pd.Timestamp(str(VAL_END),   tz=None)

    print(f"  Val calibration period: {val_start.date()} – {val_end.date()}")

    print(f"Building val rolling windows (stride={stride_hours}h)...")
    entries, actuals, _, _ = _make_rolling_instances(
        df, station_ids, station_to_idx,
        val_start, val_end,
        predictor.prediction_length,
        stride_hours,
    )
    n_windows = len(entries)
    print(f"  {n_windows} val windows")

    ds = ListDataset(entries, freq=FREQ)

    print(f"Generating {NUM_SAMPLES} samples per val window...")
    forecasts = list(predictor.predict(ds, num_samples=NUM_SAMPLES))
    samples_arr = np.stack([f.samples for f in forecasts], axis=0).astype(np.float32)
    # shape: (n_windows, NUM_SAMPLES, prediction_length)

    # Per-horizon conformal quantile: ceil((n+1)*(1-α)) / n
    n = n_windows
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)

    margins: dict[str, dict[str, float]] = {}

    print(f"\nConformal margins at {(1-alpha)*100:.0f}% target coverage (n={n}, q_level={q_level:.4f}):")
    print(f"  {'Horizon':>8}  {'upper margin':>14}  {'lower margin':>14}  {'val coverage':>13}")

    for hidx, label in zip(HORIZON_INDICES, HORIZON_LABELS):
        h_samples = samples_arr[:, :, hidx]   # (n, NUM_SAMPLES)
        h_true    = actuals[:, hidx]           # (n,)

        q05 = np.percentile(h_samples, 5,  axis=1)
        q95 = np.percentile(h_samples, 95, axis=1)

        s_upper = np.maximum(h_true - q95, 0.0)   # > 0 when above p95
        s_lower = np.maximum(q05 - h_true, 0.0)   # > 0 when below p05

        margin_upper = float(np.quantile(s_upper, q_level))
        margin_lower = float(np.quantile(s_lower, q_level))

        # Val coverage before and after adjustment (diagnostic only)
        val_cov_before = float(((h_true >= q05) & (h_true <= q95)).mean() * 100)
        val_cov_after  = float(
            ((h_true >= q05 - margin_lower) & (h_true <= q95 + margin_upper)).mean() * 100
        )

        margins[label] = {"upper": margin_upper, "lower": margin_lower}

        print(
            f"  {label:>8}  {margin_upper:>+14.3f}  {margin_lower:>+14.3f}  "
            f"{val_cov_before:>6.1f}% → {val_cov_after:.1f}%"
        )

    return margins


# ---------------------------------------------------------------------------
# Apply margins to test samples
# ---------------------------------------------------------------------------

def apply_margins(
    margins: dict[str, dict[str, float]],
    samples_path: Path = SAMPLES_PATH,
    output: Path = CONFORMAL_METRICS_OUT,
) -> dict:
    """
    Load existing test samples, apply conformal margins, recompute metrics.
    """
    print(f"\nLoading test samples from {samples_path}...")
    npz = np.load(samples_path, allow_pickle=True)
    samples_arr = npz["samples"].astype(np.float32)   # (N, NUM_SAMPLES, 72)
    actuals     = npz["actuals"].astype(np.float32)   # (N, 72)
    print(f"  {samples_arr.shape[0]} test windows")

    results: dict = {"horizons": {}, "overall": {}}
    all_mae, all_se, all_covered, all_widths, all_crps = [], [], [], [], []

    print(f"\n{'Horizon':>8}  {'MAE':>8}  {'RMSE':>8}  {'Coverage':>10}  {'Width':>8}  {'CRPS':>8}")
    for hidx, label in zip(HORIZON_INDICES, HORIZON_LABELS):
        h_samples = samples_arr[:, :, hidx]
        h_true    = actuals[:, hidx]

        q05 = np.percentile(h_samples, 5,  axis=1)
        p50 = np.percentile(h_samples, 50, axis=1)
        q95 = np.percentile(h_samples, 95, axis=1)

        m = margins[label]
        adj_q05 = q05 - m["lower"]
        adj_q95 = q95 + m["upper"]

        ae      = np.abs(p50 - h_true)
        se      = (p50 - h_true) ** 2
        covered = ((h_true >= adj_q05) & (h_true <= adj_q95)).astype(float)
        width   = adj_q95 - adj_q05
        crps    = _crps_energy(h_samples, h_true)   # CRPS unchanged (uses samples)

        mae      = float(ae.mean())
        rmse     = float(np.sqrt(se.mean()))
        coverage = float(covered.mean() * 100)
        sharpness = float(width.mean())

        results["horizons"][label] = {
            "mae": mae, "rmse": rmse,
            "pi_coverage_pct": coverage,
            "sharpness_mean_width": sharpness,
            "crps": crps,
            "conformal_margin_upper": m["upper"],
            "conformal_margin_lower": m["lower"],
        }
        all_mae.append(ae); all_se.append(se)
        all_covered.append(covered); all_widths.append(width)
        all_crps.append(crps)

        print(f"  {label:>6}  {mae:>8.3f}  {rmse:>8.3f}  {coverage:>9.1f}%  {sharpness:>8.2f}  {crps:>8.4f}")

    results["overall"] = {
        "mae":             float(np.concatenate(all_mae).mean()),
        "rmse":            float(np.sqrt(np.concatenate(all_se).mean())),
        "pi_coverage_pct": float(np.concatenate(all_covered).mean() * 100),
        "sharpness":       float(np.concatenate(all_widths).mean()),
        "crps":            float(np.mean(all_crps)),
    }
    ov = results["overall"]
    print(
        f"\n  Overall  MAE={ov['mae']:.3f}  RMSE={ov['rmse']:.3f}  "
        f"Coverage={ov['pi_coverage_pct']:.1f}%  "
        f"Width={ov['sharpness']:.2f}  CRPS={ov['crps']:.4f}"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nConformal metrics saved to {output}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformal calibration for DeepAR PIs")
    parser.add_argument("--alpha",        type=float, default=0.10,
                        help="Miscoverage level (default 0.10 → 90%% target)")
    parser.add_argument("--stride-hours", type=int,   default=STRIDE_HOURS)
    parser.add_argument("--predictor-path", type=Path, default=PREDICTOR_PATH)
    parser.add_argument("--margins-output", type=Path, default=MARGINS_OUTPUT)
    parser.add_argument("--output",         type=Path, default=CONFORMAL_METRICS_OUT)
    args = parser.parse_args()

    margins = calibrate(
        predictor_path=args.predictor_path,
        alpha=args.alpha,
        stride_hours=args.stride_hours,
    )

    args.margins_output.parent.mkdir(parents=True, exist_ok=True)
    args.margins_output.write_text(json.dumps(margins, indent=2))
    print(f"\nMargins saved to {args.margins_output}")

    apply_margins(margins, output=args.output)
