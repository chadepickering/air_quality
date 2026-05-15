"""
Step 8.4 — DeepAR Monte Carlo sample generation and test evaluation.

Usage (from project root, using venv_deepar):
    source venv_deepar/bin/activate
    python -m models.deepar.sample_forecasts [--predictor-path PATH] [--output PATH]

Loads the serialized predictor, runs rolling inference over the test split,
generates NUM_SAMPLES=500 Monte Carlo trajectories per window, and writes:

    evaluation/deepar_metrics.json   — per-horizon MAE/RMSE/PI coverage/CRPS
    evaluation/deepar_samples.npz    — raw samples for the alert system
                                       arrays: samples (N, 500, 72)
                                               actuals (N, 72)
                                               station_ids (N,) str
                                               window_starts (N,) timestamps

Rolling evaluation: one forecast every STRIDE_HOURS through the test period.
STRIDE_HOURS=24 gives ~59 windows per station × 14 stations = ~826 total windows —
enough for stable metric estimates without running 500-sample inference
at every hour of the test period.

Metrics:
    MAE / RMSE   — on p50 (median) point forecast; comparable to LSTM/TFT
    PI Coverage  — fraction of true pm25 inside [p5, p95] (target: 85–95%)
    Sharpness    — mean p5–p95 width (narrower = better given coverage)
    CRPS         — energy-form Continuous Ranked Probability Score;
                   lower = better; primary DeepAR metric
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.deepar.model import (
    CONTEXT_LENGTH,
    DYNAMIC_REAL_FEATURES,
    FREQ,
    HORIZON_INDICES,
    HORIZONS,
    NUM_SAMPLES,
    PREDICTOR_PATH,
    TARGET,
)
from models.deepar.train import load_and_prepare_df

EVAL_DIR       = Path(__file__).parent.parent.parent / "evaluation"
DEFAULT_OUTPUT = EVAL_DIR / "deepar_metrics.json"
SAMPLES_OUTPUT = EVAL_DIR / "deepar_samples.npz"

HORIZON_LABELS = [f"h{h}" for h in HORIZONS]
STRIDE_HOURS   = 24   # rolling window stride through test period


# ---------------------------------------------------------------------------
# CRPS helper
# ---------------------------------------------------------------------------

def _crps_energy(samples: np.ndarray, truth: np.ndarray) -> float:
    """
    Energy-form CRPS averaged over all provided (samples, truth) pairs.

    Args:
        samples: shape (N_windows, N_samples) — Monte Carlo draws
        truth:   shape (N_windows,)           — observed values

    CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]
    Approximated via Monte Carlo:
      term1 = mean over samples of |sample - y|  per window, then mean over windows
      term2 = mean over all pairs (i,j) of |sample_i - sample_j|, estimated
              efficiently as: std(samples) * sqrt(2/pi)  [exact for Gaussian;
              reasonable approximation for StudentT with sufficient samples]
    """
    term1 = np.mean(np.abs(samples - truth[:, None]))
    # Pairwise term via sorted-samples trick (exact, O(N log N)):
    s_sorted = np.sort(samples, axis=1)   # (N_windows, N_samples)
    n = samples.shape[1]
    weights = 2 * np.arange(1, n + 1) - n - 1   # (N_samples,)
    term2 = np.mean(np.sum(weights * s_sorted, axis=1)) / (n * (n - 1))
    return float(term1 - term2)


# ---------------------------------------------------------------------------
# Rolling window construction
# ---------------------------------------------------------------------------

def _make_rolling_instances(
    df: pd.DataFrame,
    station_ids: list[str],
    station_to_idx: dict[str, int],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    prediction_length: int,
    stride_hours: int,
) -> tuple[list[dict], np.ndarray, list[str], list[pd.Timestamp]]:
    """
    Build ListDataset entries for rolling evaluation.

    For each stride position in [test_start, test_end - prediction_length],
    create one entry per station: the series up to (and including) the context
    window before that stride position.  The next prediction_length steps are
    collected as ground-truth actuals.

    Returns:
        entries          — list of dicts ready for ListDataset
        actuals          — shape (N_windows, prediction_length) float32
        entry_station_ids — parallel list of station IDs
        window_starts    — parallel list of forecast start timestamps
    """
    from gluonts.dataset.common import ListDataset

    stride = pd.Timedelta(hours=stride_hours)
    pred_td = pd.Timedelta(hours=prediction_length)

    entries: list[dict]        = []
    actuals_list: list         = []
    entry_station_ids: list[str] = []
    window_starts: list[pd.Timestamp] = []

    for sid in station_ids:
        sdf = df[df["station_id"] == sid].sort_values("timestamp").reset_index(drop=True)

        t = test_start
        while t + pred_td <= test_end + pd.Timedelta(hours=1):
            # Context: up to (not including) t
            ctx_start = t - pd.Timedelta(hours=CONTEXT_LENGTH)
            ctx_mask  = (sdf["timestamp"] >= ctx_start) & (sdf["timestamp"] < t)
            fut_mask  = (sdf["timestamp"] >= t) & (sdf["timestamp"] < t + pred_td)

            if ctx_mask.sum() < CONTEXT_LENGTH or fut_mask.sum() < prediction_length:
                t += stride
                continue

            ctx_df = sdf[ctx_mask]
            fut_df = sdf[fut_mask]

            # GluonTS DeepAR needs feat_dynamic_real covering context + future
            # (context_length + prediction_length steps) so the InstanceSplitter
            # has decoder inputs for the prediction horizon.
            ctx_and_fut_df = sdf[ctx_mask | fut_mask].sort_values("timestamp")
            target       = ctx_df[TARGET].to_numpy(dtype=np.float32)
            feat_dynamic = ctx_and_fut_df[DYNAMIC_REAL_FEATURES].to_numpy(dtype=np.float32).T
            start        = pd.Period(ctx_df["timestamp"].iloc[0], freq=FREQ)

            entries.append({
                "start":             start,
                "target":            target,
                "feat_dynamic_real": feat_dynamic,
                "feat_static_cat":   np.array([station_to_idx[sid]], dtype=np.int32),
            })
            actuals_list.append(fut_df[TARGET].to_numpy(dtype=np.float32))
            entry_station_ids.append(sid)
            window_starts.append(t)

            t += stride

    actuals = np.stack(actuals_list, axis=0)   # (N_windows, prediction_length)
    return entries, actuals, entry_station_ids, window_starts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    predictor_path: Path = PREDICTOR_PATH,
    output: Path = DEFAULT_OUTPUT,
    samples_output: Path = SAMPLES_OUTPUT,
    stride_hours: int = STRIDE_HOURS,
) -> dict:
    from gluonts.dataset.common import ListDataset
    from gluonts.torch.model.deepar import DeepARPredictor

    # --- load predictor ---
    print(f"Loading predictor from {predictor_path}...")
    predictor = DeepARPredictor.deserialize(predictor_path)

    # --- load full series (val context + test targets) ---
    print("Loading val+test data...")
    df = load_and_prepare_df(["val", "test"])
    station_ids   = sorted(df["station_id"].unique().tolist())
    station_to_idx = {sid: i for i, sid in enumerate(station_ids)}

    from streaming.feature_engineering import VAL_END
    test_start = pd.Timestamp(str(VAL_END), tz=None) + pd.Timedelta(hours=1)
    test_end   = df["timestamp"].max()

    print(f"  {len(station_ids)} stations, test period {test_start.date()} – {test_end.date()}")

    # --- build rolling evaluation instances ---
    print(f"Building rolling windows (stride={stride_hours}h)...")
    entries, actuals, entry_sids, window_starts = _make_rolling_instances(
        df, station_ids, station_to_idx,
        test_start, test_end,
        predictor.prediction_length,
        stride_hours,
    )
    n_windows = len(entries)
    print(f"  {n_windows} windows ({n_windows // len(station_ids)} per station avg)")

    ds = ListDataset(entries, freq=FREQ)

    # --- inference ---
    print(f"Generating {NUM_SAMPLES} samples per window...")
    forecasts = list(predictor.predict(ds, num_samples=NUM_SAMPLES))

    # samples array: (N_windows, NUM_SAMPLES, prediction_length)
    samples_arr = np.stack(
        [f.samples for f in forecasts], axis=0
    ).astype(np.float32)   # (N, 500, 72)

    # --- per-horizon metrics ---
    results: dict = {"horizons": {}, "overall": {}}
    all_mae, all_se, all_covered, all_widths, all_crps = [], [], [], [], []

    for hidx, label in zip(HORIZON_INDICES, HORIZON_LABELS):
        h_samples = samples_arr[:, :, hidx]   # (N, 500)
        h_true    = actuals[:, hidx]           # (N,)

        p5  = np.percentile(h_samples, 5,  axis=1)
        p50 = np.percentile(h_samples, 50, axis=1)
        p95 = np.percentile(h_samples, 95, axis=1)

        ae      = np.abs(p50 - h_true)
        se      = (p50 - h_true) ** 2
        covered = ((h_true >= p5) & (h_true <= p95)).astype(float)
        width   = p95 - p5
        crps    = _crps_energy(h_samples, h_true)

        mae      = float(ae.mean())
        rmse     = float(np.sqrt(se.mean()))
        coverage = float(covered.mean() * 100)
        sharpness = float(width.mean())

        results["horizons"][label] = {
            "mae": mae, "rmse": rmse,
            "pi_coverage_pct": coverage,
            "sharpness_mean_width": sharpness,
            "crps": crps,
        }
        all_mae.append(ae); all_se.append(se)
        all_covered.append(covered); all_widths.append(width)
        all_crps.append(crps)

        print(
            f"  {label:>4s}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
            f"Coverage={coverage:.1f}%  Width={sharpness:.2f}  CRPS={crps:.4f}"
        )

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

    # --- save outputs ---
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics saved to {output}")

    np.savez(
        samples_output,
        samples=samples_arr,
        actuals=actuals,
        station_ids=np.array(entry_sids),
        window_starts=np.array([str(t) for t in window_starts]),
    )
    print(f"Samples saved to {samples_output}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepAR sample generation and evaluation")
    parser.add_argument("--predictor-path", type=Path, default=PREDICTOR_PATH)
    parser.add_argument("--output",         type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-output", type=Path, default=SAMPLES_OUTPUT)
    parser.add_argument("--stride-hours",   type=int,  default=STRIDE_HOURS)
    args = parser.parse_args()

    evaluate(
        predictor_path=args.predictor_path,
        output=args.output,
        samples_output=args.samples_output,
        stride_hours=args.stride_hours,
    )
