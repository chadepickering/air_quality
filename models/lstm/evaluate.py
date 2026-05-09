"""
Step 6.5 — LSTM evaluation on the held-out test split.

Usage:
    python -m models.lstm.evaluate [--checkpoint PATH] [--output PATH]

Loads the best checkpoint and scaler saved by train.py, runs inference on
the test split, and writes per-horizon MAE / RMSE / MAPE to a JSON file.

Output written to evaluation/lstm_metrics.json by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lstm.model import HORIZONS, LSTMForecaster
from models.lstm.train import (
    AQDataset,
    CHECKPOINT_PATH,
    N_FEATURES,
    SCALER_PATH,
    SEQ_LEN,
    build_windows,
    load_scaler,
    load_split_df,
)

EVAL_DIR = Path(__file__).parent.parent.parent / "evaluation"
DEFAULT_OUTPUT = EVAL_DIR / "lstm_metrics.json"

HORIZON_LABELS = [f"h{h}" for h in HORIZONS]


def _mape(pred: np.ndarray, true: np.ndarray, eps: float = 1.0) -> float:
    """MAPE clamped so near-zero true values don't inflate the metric."""
    denom = np.maximum(np.abs(true), eps)
    return float(np.mean(np.abs(pred - true) / denom) * 100)


def evaluate(
    checkpoint: Path = CHECKPOINT_PATH,
    scaler_path: Path = SCALER_PATH,
    output: Path = DEFAULT_OUTPUT,
    batch_size: int = 512,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model ---
    model = LSTMForecaster(n_features=N_FEATURES)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # --- load scaler and test data ---
    mean, std = load_scaler(scaler_path)

    print("Loading test split...")
    test_df = load_split_df("test")
    print(f"  {len(test_df):,} rows, {test_df['station_id'].nunique()} stations")

    # Prepend tail of val data so windows near the split boundary have context.
    import duckdb
    from ingestion.database import DB_PATH
    from models.lstm.train import FEATURE_COLS
    from streaming.feature_engineering import VAL_END
    con = duckdb.connect(str(DB_PATH), read_only=True)
    boundary_df = con.execute(f"""
        SELECT station_id, timestamp, {', '.join(FEATURE_COLS)}
        FROM processed_features
        WHERE split = 'val'
          AND timestamp >= TIMESTAMPTZ '{VAL_END}' - INTERVAL '96 hours'
        ORDER BY station_id, timestamp
    """).df()
    con.close()
    import pandas as pd
    boundary_df["timestamp"] = pd.to_datetime(boundary_df["timestamp"])
    test_df = pd.concat([boundary_df, test_df], ignore_index=True)
    test_df = test_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    print("Building test windows...")
    X_te, y_te = build_windows(test_df, mean, std)
    print(f"  {len(X_te):,} test windows")

    loader = DataLoader(AQDataset(X_te, y_te),
                        batch_size=batch_size, shuffle=False, num_workers=0)

    # --- inference ---
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            all_preds.append(model(xb.to(device)).cpu())
            all_targets.append(yb)

    preds   = torch.cat(all_preds).numpy()    # (N, 4)
    targets = torch.cat(all_targets).numpy()  # (N, 4)

    # --- per-horizon metrics ---
    results: dict = {"horizons": {}, "overall": {}}
    all_ae, all_se = [], []

    for i, label in enumerate(HORIZON_LABELS):
        p, t = preds[:, i], targets[:, i]
        ae = np.abs(p - t)
        se = (p - t) ** 2
        mae  = float(ae.mean())
        rmse = float(np.sqrt(se.mean()))
        mape = _mape(p, t)
        results["horizons"][label] = {"mae": mae, "rmse": rmse, "mape": mape}
        all_ae.append(ae)
        all_se.append(se)
        print(f"  {label:>4s}  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%")

    results["overall"] = {
        "mae":  float(np.concatenate(all_ae).mean()),
        "rmse": float(np.sqrt(np.concatenate(all_se).mean())),
    }
    print(f"\n  Overall MAE={results['overall']['mae']:.3f}  "
          f"RMSE={results['overall']['rmse']:.3f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics saved to {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM on test split")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output",     type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int,  default=512)
    args = parser.parse_args()

    evaluate(checkpoint=args.checkpoint, output=args.output, batch_size=args.batch_size)
