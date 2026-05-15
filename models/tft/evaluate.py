"""
Step 7.4 — TFT evaluation on the held-out test split.

Usage:
    python -m models.tft.evaluate [--checkpoint PATH] [--output PATH]

Loads the best Lightning checkpoint and dataset parameters saved by train.py,
runs rolling inference across the full test split, and writes per-horizon
metrics to JSON.

Rolling evaluation:
    All windows where the 72-step decoder falls within the test period are
    evaluated.  The val period supplies encoder context for windows near the
    test boundary.  This gives ~2,000+ windows per station × 14 stations,
    comparable in scale to LSTM and DeepAR evaluations.

Metrics computed:
    MAE / RMSE  — on p50 (median) point forecast; compared directly to LSTM
    PI Coverage — fraction of true pm25 values inside [p5, p95] (target: 85-95%)
    Sharpness   — mean width of p5-p95 interval (narrower = better given coverage)

Output written to evaluation/tft_metrics.json by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.tft.model import (
    HORIZON_INDICES,
    HORIZONS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    QUANTILES,
)
from models.tft.train import (
    CKPT_PATH,
    DATASET_PATH,
    load_and_prepare_df,
)
from streaming.feature_engineering import VAL_END

EVAL_DIR       = Path(__file__).parent.parent.parent / "evaluation"
DEFAULT_OUTPUT = EVAL_DIR / "tft_metrics.json"

FRM_ONLY_STATIONS = {
    "06-037-1201", "06-037-1602", "06-037-2005",
    "06-071-2002", "06-071-9004",
}

HORIZON_LABELS = [f"h{h}" for h in HORIZONS]
P5_IDX  = QUANTILES.index(0.05)
P50_IDX = QUANTILES.index(0.5)
P95_IDX = QUANTILES.index(0.95)


def evaluate(
    checkpoint: Path = CKPT_PATH,
    dataset_params: Path = DATASET_PATH,
    output: Path = DEFAULT_OUTPUT,
    batch_size: int = 256,
) -> dict:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

    # Test period starts one hour after val ends.
    # Include enough val context (MAX_ENCODER_LENGTH hours) so the first
    # test-period window has a full encoder lookback.
    test_start    = pd.Timestamp(str(VAL_END), tz=None) + pd.Timedelta(hours=1)
    encoder_start = test_start - pd.Timedelta(hours=MAX_ENCODER_LENGTH)

    # --- load data: val+test, FRM stations excluded, trimmed to encoder context ---
    print("Loading val+test data...")
    df = load_and_prepare_df(["val", "test"])
    df = df[~df["station_id"].isin(FRM_ONLY_STATIONS)].copy()
    df = df[df["timestamp"] >= encoder_start].reset_index(drop=True)
    print(f"  {len(df):,} rows ({df['station_id'].nunique()} stations, "
          f"from {df['timestamp'].min().date()} to {df['timestamp'].max().date()})")

    params  = torch.load(dataset_params, weights_only=False)
    dataset = TimeSeriesDataSet.from_parameters(params, df, predict=False)
    loader  = dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )
    print(f"  {len(dataset):,} rolling prediction windows")

    # --- load model ---
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # --- inference ---
    # return_y=True makes pytorch-forecasting include the targets in the original
    # (denormalised) PM2.5 scale alongside the quantile predictions.
    raw_preds = model.predict(
        loader,
        mode="quantiles",
        return_index=True,
        return_decoder_lengths=True,
        return_y=True,
    )

    # raw_preds.output: (N, 72, 3) — quantile predictions in original PM2.5 scale
    # raw_preds.y[0]:   (N, 72)    — actuals in original PM2.5 scale
    preds_np   = raw_preds.output.cpu().numpy()   # (N, 72, 3)
    actuals_np = raw_preds.y[0].cpu().numpy()     # (N, 72)

    n_windows = preds_np.shape[0]
    print(f"\nComputing metrics over {n_windows:,} windows × 72 steps...")

    # --- per-horizon metrics ---
    results: dict = {"n_windows": n_windows, "horizons": {}, "overall": {}}
    all_ae_p50, all_se_p50 = [], []
    all_covered, all_widths = [], []

    for hidx, label in zip(HORIZON_INDICES, HORIZON_LABELS):
        p5   = preds_np[:, hidx, P5_IDX]
        p50  = preds_np[:, hidx, P50_IDX]
        p95  = preds_np[:, hidx, P95_IDX]
        true = actuals_np[:, hidx]

        ae      = np.abs(p50 - true)
        se      = (p50 - true) ** 2
        covered = ((true >= p5) & (true <= p95)).astype(float)
        width   = p95 - p5

        mae       = float(ae.mean())
        rmse      = float(np.sqrt(se.mean()))
        coverage  = float(covered.mean() * 100)
        sharpness = float(width.mean())

        results["horizons"][label] = {
            "mae": mae, "rmse": rmse,
            "pi_coverage_pct": coverage,
            "sharpness_mean_width": sharpness,
        }
        all_ae_p50.append(ae)
        all_se_p50.append(se)
        all_covered.append(covered)
        all_widths.append(width)

        print(
            f"  {label:>4s}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
            f"Coverage={coverage:.1f}%  Width={sharpness:.2f} μg/m³"
        )

    results["overall"] = {
        "mae":             float(np.concatenate(all_ae_p50).mean()),
        "rmse":            float(np.sqrt(np.concatenate(all_se_p50).mean())),
        "pi_coverage_pct": float(np.concatenate(all_covered).mean() * 100),
        "sharpness":       float(np.concatenate(all_widths).mean()),
    }
    ov = results["overall"]
    print(
        f"\n  Overall  MAE={ov['mae']:.3f}  RMSE={ov['rmse']:.3f}  "
        f"Coverage={ov['pi_coverage_pct']:.1f}%  "
        f"Width={ov['sharpness']:.2f} μg/m³"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics saved to {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TFT on test split")
    parser.add_argument("--checkpoint",     type=Path, default=CKPT_PATH)
    parser.add_argument("--dataset-params", type=Path, default=DATASET_PATH)
    parser.add_argument("--output",         type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size",     type=int,  default=256)
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        dataset_params=args.dataset_params,
        output=args.output,
        batch_size=args.batch_size,
    )
