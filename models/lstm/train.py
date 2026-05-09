"""
Step 6.4 — LSTM training script.

Usage:
    python -m models.lstm.train [options]

    --epochs     Max training epochs (default: 50)
    --batch-size Mini-batch size     (default: 256)
    --lr         Initial learning rate (default: 1e-3)
    --patience   Early-stopping patience in epochs (default: 5)
    --no-wandb   Disable W&B logging (log to console only)

Outputs (written to models/lstm/):
    scaler.npz        — z-score mean/std fit on train split (reused by evaluate.py)
    best_model.pt     — state dict of the checkpoint with lowest val MAE
    train_metrics.json — final epoch train/val metrics
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ingestion.database import DB_PATH
from models.lstm.model import HORIZONS, LSTMForecaster
from streaming.feature_engineering import TRAIN_END, VAL_END

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "pm25", "no2", "o3", "pm10", "co",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "pm25_roll3", "pm25_roll6", "pm25_roll24",
    "pm25_lag1", "pm25_lag3", "pm25_lag24",
    "spatial_pm25_lag1", "spatial_pm25_lag3", "spatial_pm25_roll6",
    "spatial_no2_lag1", "spatial_o3_lag1", "spatial_elev_diff",
]
TARGET_COL = "pm25"
SEQ_LEN    = 24   # 24hr lookback window

MODEL_DIR  = Path(__file__).parent
SCALER_PATH      = MODEL_DIR / "scaler.npz"
CHECKPOINT_PATH  = MODEL_DIR / "best_model.pt"
METRICS_PATH     = MODEL_DIR / "train_metrics.json"

N_FEATURES = len(FEATURE_COLS)


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit z-score params on X (rows × features), ignoring NaN."""
    mean = np.nanmean(X, axis=0)
    std  = np.nanstd(X,  axis=0)
    std  = np.where(std == 0, 1.0, std)   # avoid divide-by-zero for constant cols
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalize X, then fill remaining NaN with 0 (scaled mean)."""
    X = (X - mean) / std
    np.nan_to_num(X, nan=0.0, copy=False)
    return X


def save_scaler(mean: np.ndarray, std: np.ndarray, path: Path = SCALER_PATH) -> None:
    np.savez(path, mean=mean, std=std)


def load_scaler(path: Path = SCALER_PATH) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["mean"], data["std"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split_df(split: str) -> pd.DataFrame:
    """Load processed_features for one split, sorted by (station_id, timestamp)."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT station_id, timestamp, {', '.join(FEATURE_COLS)}
        FROM processed_features
        WHERE split = '{split}'
        ORDER BY station_id, timestamp
    """).df()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------

def build_windows(
    df: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window arrays for all stations in df.

    For each station, step through timestamps. Window at position i uses the
    24 rows ending at i as input and pm25 at positions [i+h for h in HORIZONS]
    as targets.  Windows where any target pm25 is NaN are skipped.

    Returns:
        X: float32 array of shape (N, SEQ_LEN, N_FEATURES)
        y: float32 array of shape (N, 4)
    """
    df = df.reset_index(drop=True)   # guarantee 0-based index for positional slicing below
    raw = df[FEATURE_COLS].values.astype(np.float32)
    scaled = apply_scaler(raw.copy(), mean, std)

    X_list, y_list = [], []
    max_horizon = max(HORIZONS)

    for _, sdf in df.groupby("station_id", sort=False):
        idx   = sdf.index.values          # positions in df
        pm25  = sdf[TARGET_COL].values    # raw (unscaled) targets
        feat  = scaled[idx]               # pre-scaled feature rows

        n = len(idx)
        for i in range(SEQ_LEN, n - max_horizon):
            window = feat[i - SEQ_LEN : i]   # (SEQ_LEN, N_FEATURES)

            # targets are raw pm25 values (not scaled — we evaluate in μg/m³)
            targets = np.array([pm25[i + h] for h in HORIZONS], dtype=np.float32)
            if np.any(np.isnan(targets)):
                continue

            X_list.append(window)
            y_list.append(targets)

    if not X_list:
        return np.empty((0, SEQ_LEN, N_FEATURES), dtype=np.float32), \
               np.empty((0, len(HORIZONS)), dtype=np.float32)

    return np.stack(X_list), np.stack(y_list)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AQDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs:     int   = 50,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    patience:   int   = 5,
    use_wandb:  bool  = True,
) -> dict:
    """
    Full training run.  Returns final metrics dict.

    Leakage control:
        - Scaler fitted on train rows only, then applied to val.
        - Val windows may look back 24hr into train data — this is fine;
          the features are already computed without future information.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data ---
    print("Loading train split...")
    t0 = time.time()
    train_df = load_split_df("train")
    print(f"  {len(train_df):,} rows in {time.time()-t0:.1f}s")

    print("Fitting scaler on train features...")
    mean, std = fit_scaler(train_df[FEATURE_COLS].values.astype(np.float32))
    save_scaler(mean, std)

    print("Building train windows...")
    X_tr, y_tr = build_windows(train_df, mean, std)
    print(f"  {len(X_tr):,} train windows")
    del train_df   # free memory before loading val

    print("Loading val split and building windows...")
    val_df = load_split_df("val")

    # Prepend the last SEQ_LEN+max(HORIZONS) rows of train to each station's
    # val data so val windows that start near the split boundary have context.
    con = duckdb.connect(str(DB_PATH), read_only=True)
    boundary_df = con.execute(f"""
        SELECT station_id, timestamp, {', '.join(FEATURE_COLS)}
        FROM processed_features
        WHERE split = 'train'
          AND timestamp >= TIMESTAMPTZ '{TRAIN_END}' - INTERVAL '96 hours'
        ORDER BY station_id, timestamp
    """).df()
    con.close()
    boundary_df["timestamp"] = pd.to_datetime(boundary_df["timestamp"])
    val_df = pd.concat([boundary_df, val_df], ignore_index=True)
    val_df = val_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    X_val, y_val = build_windows(val_df, mean, std)
    print(f"  {len(X_val):,} val windows")
    del val_df, boundary_df

    train_loader = DataLoader(AQDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(AQDataset(X_val, y_val),
                              batch_size=batch_size * 2, shuffle=False,
                              num_workers=0)

    # --- model ---
    model = LSTMForecaster(n_features=N_FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # --- W&B ---
    if use_wandb:
        try:
            import wandb
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "air-quality-forecasting"),
                name="lstm-baseline",
                config={
                    "model": "LSTM",
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "seq_len": SEQ_LEN,
                    "horizons": HORIZONS,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "patience": patience,
                    "n_features": N_FEATURES,
                    "n_train_windows": len(X_tr),
                    "n_val_windows": len(X_val),
                },
            )
        except Exception as e:
            print(f"W&B init failed ({e}); continuing without tracking.")
            use_wandb = False

    # --- training loop ---
    best_val_mae = float("inf")
    patience_counter = 0

    horizon_labels = [f"h{h}" for h in HORIZONS]

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # val
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                all_preds.append(model(xb.to(device)).cpu())
                all_targets.append(yb)
        preds   = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        per_horizon_mae = np.abs(preds - targets).mean(axis=0)
        val_mae = per_horizon_mae.mean()

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_mae,
            **{f"val_mae_{horizon_labels[i]}": float(per_horizon_mae[i])
               for i in range(len(HORIZONS))},
            "lr": scheduler.get_last_lr()[0],
        }
        if use_wandb:
            import wandb
            wandb.log(log)

        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_mae={val_mae:.3f} μg/m³ | "
            f"[{', '.join(f'{m:.2f}' for m in per_horizon_mae)}]"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✓ New best checkpoint saved (val_mae={best_val_mae:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    metrics = {
        "best_val_mae": float(best_val_mae),
        "stopped_at_epoch": epoch,
        "per_horizon_val_mae": {horizon_labels[i]: float(per_horizon_mae[i])
                                for i in range(len(HORIZONS))},
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nTraining complete. Best val MAE: {best_val_mae:.3f} μg/m³")
    print(f"Metrics saved to {METRICS_PATH}")

    if use_wandb:
        import wandb
        wandb.finish()

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM PM2.5 forecaster")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--no-wandb",   action="store_true")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        use_wandb=not args.no_wandb,
    )
