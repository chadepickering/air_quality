"""
Step 6.6 — λ / d_cutoff grid search on the validation set.

Strategy:
    Load processed_features from DuckDB once. For each (λ, d_cutoff) pair,
    recompute only the six spatial columns in-memory (all other features are
    unchanged), rebuild windows, train the LSTM for N_SEARCH_EPOCHS epochs,
    and record val MAE.  No DuckDB writes; the search is fully in-memory.

    After the search, the best parameters are printed and written to
    models/lstm/lambda_search_results.json.  The caller is expected to run
    the full training script (train.py) after updating station_registry.py
    with the optimal λ and d_cutoff.

Usage:
    python -m models.lstm.lambda_search [--search-epochs N]
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ingestion.database import DB_PATH
from ingestion.station_registry import (
    build_spatial_neighbor_index,
    load_stations,
)
from models.lstm.model import HORIZONS, LSTMForecaster
from models.lstm.train import (
    AQDataset,
    FEATURE_COLS,
    N_FEATURES,
    SEQ_LEN,
    SCALER_PATH,
    build_windows,
    fit_scaler,
    save_scaler,
    load_scaler,
    apply_scaler,
)
from streaming.feature_engineering import TRAIN_END, PARAMETERS
from streaming.spatial_weights import compute_spatial_features

RESULTS_PATH = Path(__file__).parent / "lambda_search_results.json"

# Search grid (λ in km²/m²; see station_registry.py for unit explanation)
LAMBDA_GRID   = [0.0001, 0.0005, 0.001]
D_CUTOFF_GRID = [30.0, 40.0, 50.0]

N_SEARCH_EPOCHS = 15   # proxy training — enough signal, not full convergence
BATCH_SIZE      = 256
LR              = 1e-3

SPATIAL_COLS = [
    "spatial_pm25_lag1", "spatial_pm25_lag3", "spatial_pm25_roll6",
    "spatial_no2_lag1",  "spatial_o3_lag1",   "spatial_elev_diff",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_base_df() -> pd.DataFrame:
    """Load processed_features for train+val, sorted by (station_id, timestamp)."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT station_id, timestamp, {', '.join(FEATURE_COLS)}, split
        FROM processed_features
        WHERE split IN ('train', 'val')
        ORDER BY station_id, timestamp
    """).df()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _recompute_spatial(
    df: pd.DataFrame,
    stations: list[dict],
    lambda_param: float,
    d_cutoff: float,
) -> pd.DataFrame:
    """
    Replace the six spatial columns in df using the given λ and d_cutoff.

    Pivots df to per-station DataFrames (indexed by timestamp), runs
    compute_spatial_features for each station, then merges back.
    """
    neighbor_index = build_spatial_neighbor_index(stations, lambda_param, d_cutoff)
    elevation_lookup = {
        str(s["station_id"]): float(s.get("elevation_m") or 0.0)
        for s in stations
    }

    # Build per-station DataFrames on timestamp index (needed by spatial_weights)
    station_dfs: dict[str, pd.DataFrame] = {}
    for sid, sdf in df.groupby("station_id", sort=False):
        sdf = sdf.set_index("timestamp").sort_index()
        for col in PARAMETERS:
            if col not in sdf.columns:
                sdf[col] = np.nan
        station_dfs[str(sid)] = sdf

    # Recompute spatial features per station
    updated: list[pd.DataFrame] = []
    for sid, sdf in station_dfs.items():
        neighbors = neighbor_index.get(sid, [])
        sdf = compute_spatial_features(
            sid, sdf, neighbors, station_dfs, elevation_lookup
        )
        sdf = sdf.reset_index()   # timestamp index → column
        sdf["station_id"] = sid
        updated.append(sdf)

    updated_df = pd.concat(updated, ignore_index=True)

    # Merge spatial columns back into original df (drop old spatial cols first)
    non_spatial = [c for c in df.columns if c not in SPATIAL_COLS]
    result = df[non_spatial].merge(
        updated_df[["station_id", "timestamp"] + SPATIAL_COLS],
        on=["station_id", "timestamp"],
        how="left",
    )
    return result.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick-train helper
# ---------------------------------------------------------------------------

def _quick_train_eval(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int,
    device: torch.device,
) -> float:
    """Train a fresh LSTM for n_epochs; return mean val MAE across horizons."""
    model = LSTMForecaster(n_features=N_FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_loader = DataLoader(AQDataset(X_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(AQDataset(X_val, y_val),
                              batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds.append(model(xb.to(device)).cpu())
            targets.append(yb)
    p = torch.cat(preds).numpy()
    t = torch.cat(targets).numpy()
    return float(np.abs(p - t).mean())


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def lambda_search(n_search_epochs: int = N_SEARCH_EPOCHS) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Grid: λ={LAMBDA_GRID}  d_cutoff={D_CUTOFF_GRID}km  epochs={n_search_epochs}")

    print("Loading processed_features (train+val)...")
    base_df = _load_base_df()
    stations = load_stations()
    print(f"  {len(base_df):,} rows, {base_df['station_id'].nunique()} stations")

    # Fit scaler on train split (same as train.py — prevents data leakage)
    train_base = base_df[base_df["split"] == "train"]
    mean, std = fit_scaler(train_base[FEATURE_COLS].values.astype(np.float32))
    save_scaler(mean, std)   # overwrite scaler for consistency with full train run

    # Prepend tail of train data to val for boundary context (mirrors train.py)
    boundary = train_base[
        train_base["timestamp"] >= (
            pd.Timestamp(str(TRAIN_END)) - pd.Timedelta(hours=96)
        )
    ]

    results: list[dict] = []
    combos = list(product(LAMBDA_GRID, D_CUTOFF_GRID))

    for i, (lam, d_cut) in enumerate(combos, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(combos)}] λ={lam}  d_cutoff={d_cut}km")

        # Recompute spatial features for this combo
        df_spatial = _recompute_spatial(base_df, stations, lam, d_cut)

        train_df = df_spatial[df_spatial["split"] == "train"].copy()
        val_df   = pd.concat([
            boundary.drop(columns=["split"]).assign(split="val"),
            df_spatial[df_spatial["split"] == "val"],
        ], ignore_index=True)
        val_df = val_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

        X_tr,  y_tr  = build_windows(train_df, mean, std)
        X_val, y_val = build_windows(val_df,   mean, std)
        print(f"  train windows={len(X_tr):,}  val windows={len(X_val):,}")

        val_mae = _quick_train_eval(X_tr, y_tr, X_val, y_val, n_search_epochs, device)
        elapsed = time.time() - t0
        print(f"  val_mae={val_mae:.4f} μg/m³  ({elapsed:.0f}s)")

        results.append({"lambda": lam, "d_cutoff": d_cut, "val_mae": val_mae})

    # Sort by val MAE
    results.sort(key=lambda r: r["val_mae"])
    best = results[0]

    print("\n--- Grid search results (sorted by val MAE) ---")
    print(f"{'λ':>10}  {'d_cutoff':>10}  {'val_mae':>10}")
    for r in results:
        marker = " ← best" if r is best else ""
        print(f"{r['lambda']:>10}  {r['d_cutoff']:>10.0f}  {r['val_mae']:>10.4f}{marker}")

    print(f"\nBest: λ={best['lambda']}  d_cutoff={best['d_cutoff']}km  "
          f"val_mae={best['val_mae']:.4f} μg/m³")
    print(f"Update LAMBDA_DEFAULT and D_CUTOFF_KM in ingestion/station_registry.py, "
          f"then re-run python -m streaming.feature_engineering and python -m models.lstm.train")

    output = {"best": best, "all_results": results}
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"Results saved to {RESULTS_PATH}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="λ / d_cutoff grid search for spatial features")
    parser.add_argument("--search-epochs", type=int, default=N_SEARCH_EPOCHS,
                        help="Training epochs per grid point (default: 15)")
    args = parser.parse_args()

    lambda_search(n_search_epochs=args.search_epochs)
