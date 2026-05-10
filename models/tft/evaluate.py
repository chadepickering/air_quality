"""
Step 7.4 — TFT evaluation on the held-out test split.

Usage:
    python -m models.tft.evaluate [--checkpoint PATH] [--output PATH]

Loads the best Lightning checkpoint and dataset parameters saved by train.py,
runs inference on the test split, and writes per-horizon metrics to JSON.

Metrics computed:
    MAE / RMSE  — on p50 (median) point forecast; compared directly to LSTM
    PI Coverage — fraction of true pm25 values inside [p5, p95] (target: 85–95%)
    Sharpness   — mean width of p5–p95 interval (narrower = better given coverage)

Output written to evaluation/tft_metrics.json by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from ingestion.database import DB_PATH
from models.tft.model import (
    HORIZON_INDICES,
    HORIZONS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    QUANTILES,
    TARGET,
    build_tft,
)
from models.tft.train import (
    CKPT_PATH,
    DATASET_PATH,
    MODEL_DIR,
    load_and_prepare_df,
)

EVAL_DIR       = Path(__file__).parent.parent.parent / "evaluation"
DEFAULT_OUTPUT = EVAL_DIR / "tft_metrics.json"

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

    # --- rebuild dataset (val context + test rows) ---
    print("Loading val+test data...")
    df = load_and_prepare_df(["val", "test"])
    print(f"  {len(df):,} rows")

    params = torch.load(dataset_params, weights_only=False)
    dataset = TimeSeriesDataSet.from_parameters(params, df, predict=True)
    loader  = dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )
    print(f"  {len(dataset)} prediction windows")

    # --- load model ---
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # --- inference ---
    # predict() returns a tensor of shape (n_windows, max_prediction_length, n_quantiles)
    raw_preds = model.predict(
        loader,
        mode="quantiles",
        return_index=True,
        return_decoder_lengths=True,
        show_progress_bar=True,
    )

    # pytorch-forecasting returns a named tuple; unpack predictions and actuals
    predictions = raw_preds.output          # (N, 72, 3)
    actuals      = torch.cat([y[0] for _, y in loader])  # (N, 72) — scaled targets

    # Inverse-transform targets using the dataset's normaliser
    # The GroupNormalizer stored in the dataset handles per-station denormalisation.
    # We use the dataset's inverse_transform for the actuals.
    # For predictions, pytorch-forecasting's predict(mode='quantiles') already
    # returns values in the original (denormalised) scale.
    target_scale = torch.cat([
        x["target_scale"] for x, _ in loader
    ])  # (N, 2) — [center, scale] per window

    # Denormalise actuals (softplus GroupNormalizer: y_orig = softplus(y_scaled) * scale + center)
    # pytorch-forecasting provides a helper on the normalizer stored in the dataset
    normalizer = dataset.target_normalizer
    actuals_denorm = normalizer.inverse_transform(
        actuals.unsqueeze(-1), target_scale
    ).squeeze(-1)   # (N, 72)

    preds_np   = predictions.numpy()        # (N, 72, 3)
    actuals_np = actuals_denorm.numpy()     # (N, 72)

    # --- per-horizon metrics ---
    results: dict = {"horizons": {}, "overall": {}}
    all_ae_p50, all_se_p50 = [], []
    all_covered, all_widths = [], []

    for i, (hidx, label) in enumerate(zip(HORIZON_INDICES, HORIZON_LABELS)):
        p5  = preds_np[:, hidx, P5_IDX]
        p50 = preds_np[:, hidx, P50_IDX]
        p95 = preds_np[:, hidx, P95_IDX]
        true = actuals_np[:, hidx]

        ae   = np.abs(p50 - true)
        se   = (p50 - true) ** 2
        covered = ((true >= p5) & (true <= p95)).astype(float)
        width   = p95 - p5

        mae      = float(ae.mean())
        rmse     = float(np.sqrt(se.mean()))
        coverage = float(covered.mean() * 100)
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
        "mae":              float(np.concatenate(all_ae_p50).mean()),
        "rmse":             float(np.sqrt(np.concatenate(all_se_p50).mean())),
        "pi_coverage_pct":  float(np.concatenate(all_covered).mean() * 100),
        "sharpness":        float(np.concatenate(all_widths).mean()),
    }
    print(
        f"\n  Overall  MAE={results['overall']['mae']:.3f}  "
        f"RMSE={results['overall']['rmse']:.3f}  "
        f"Coverage={results['overall']['pi_coverage_pct']:.1f}%  "
        f"Width={results['overall']['sharpness']:.2f} μg/m³"
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
