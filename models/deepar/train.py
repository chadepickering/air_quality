"""
Step 8.3 — DeepAR data preparation and training script.

Usage (from project root, using venv_deepar):
    source venv_deepar/bin/activate
    python -m models.deepar.train [--epochs N] [--no-wandb]

Outputs (written to models/deepar/):
    predictor/   — serialized GluonTS predictor directory

Key design decisions vs TFT:
  - 14 stations only: FRM-only sites (5 stations, 0% hourly PM2.5) excluded.
    All three models use the same 14 stations for fair comparison.
  - ListDataset: one entry per station — GluonTS's native format. Each entry
    contains the full PM2.5 target series plus 20 dynamic real covariates.
  - train/val split: train entries end at TRAIN_END; val entries extend into
    the val period so GluonTS can compute val_loss via Lightning.
  - EarlyStopping on val_loss (patience=5) — same as TFT.
  - num_batches_per_epoch=100 (set in model.py): GluonTS stochastic batching
    makes each epoch ~30s rather than ~100min as in TFT's full-sweep approach.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from ingestion.database import DB_PATH
from models.deepar.model import (
    DYNAMIC_REAL_FEATURES,
    FREQ,
    MODEL_DIR,
    NUM_FEAT_DYNAMIC_REAL,
    TARGET,
    build_estimator,
)
from streaming.feature_engineering import TRAIN_END, VAL_END

PREDICTOR_PATH = MODEL_DIR / "predictor"
ALL_FEATURE_COLS = DYNAMIC_REAL_FEATURES + [TARGET]

# Stations excluded: 0% hourly PM2.5 coverage (FRM-only, no continuous monitor)
FRM_ONLY_STATIONS = {
    "06-037-1201", "06-037-1602", "06-037-2005",
    "06-071-2002", "06-071-9004",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare_df(splits: list[str]) -> pd.DataFrame:
    """
    Load processed_features for the 14 valid stations and given splits.

    NaN-fill strategy (within each station):
        1. Forward-fill — handles lag/roll NaN mid-series.
        2. Backward-fill — handles NaN at record start (lag features).
        3. Fill remaining with 0 — fallback for fully-null columns.
    """
    split_list = ", ".join(f"'{s}'" for s in splits)
    cols = ["station_id", "timestamp"] + ALL_FEATURE_COLS + ["split"]
    excluded = ", ".join(f"'{s}'" for s in FRM_ONLY_STATIONS)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT {', '.join(cols)}
        FROM processed_features
        WHERE split IN ({split_list})
          AND station_id NOT IN ({excluded})
        ORDER BY station_id, timestamp
    """).df()
    con.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df[ALL_FEATURE_COLS] = (
        df.groupby("station_id", group_keys=False)[ALL_FEATURE_COLS]
        .apply(lambda g: g.ffill().bfill().fillna(0.0))
    )

    return df


def _build_list_dataset(
    df: pd.DataFrame,
    station_ids: list[str],
    station_to_idx: dict[str, int],
) -> "ListDataset":
    """Convert a DataFrame to a GluonTS ListDataset (one entry per station)."""
    from gluonts.dataset.common import ListDataset

    entries = []
    for sid in station_ids:
        sdf = df[df["station_id"] == sid].sort_values("timestamp")
        if sdf.empty:
            continue

        target = sdf[TARGET].to_numpy(dtype=np.float32)
        feat_dynamic = sdf[DYNAMIC_REAL_FEATURES].to_numpy(dtype=np.float32).T  # (NUM_FEAT_DYNAMIC_REAL, T)
        start = pd.Period(sdf["timestamp"].iloc[0], freq=FREQ)

        entries.append({
            "start":             start,
            "target":            target,
            "feat_dynamic_real": feat_dynamic,
            "feat_static_cat":   np.array([station_to_idx[sid]], dtype=np.int32),
        })

    return ListDataset(entries, freq=FREQ)


def build_datasets(df: pd.DataFrame) -> tuple:
    """
    Split df into train and val ListDatasets.

    Train entries: timesteps up to and including TRAIN_END.
    Val entries:   full series (train + val period) so Lightning can compute
                   val_loss on the held-out val horizon.
    """
    train_cutoff = pd.Timestamp(str(TRAIN_END), tz=None)
    val_cutoff   = pd.Timestamp(str(VAL_END),   tz=None)

    station_ids   = sorted(df["station_id"].unique().tolist())
    station_to_idx = {sid: i for i, sid in enumerate(station_ids)}

    train_df = df[df["timestamp"] <= train_cutoff].copy()
    val_df   = df[df["timestamp"] <= val_cutoff].copy()

    train_ds = _build_list_dataset(train_df, station_ids, station_to_idx)
    val_ds   = _build_list_dataset(val_df,   station_ids, station_to_idx)

    return train_ds, val_ds, station_ids, station_to_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs:    int  = 50,
    use_wandb: bool = True,
    patience:  int  = 10,
) -> None:
    from lightning.pytorch.callbacks import EarlyStopping

    print("Loading and preparing data...")
    df = load_and_prepare_df(["train", "val"])
    n_stations = df["station_id"].nunique()
    print(f"  {len(df):,} rows, {n_stations} stations")

    print("Building ListDatasets...")
    train_ds, val_ds, station_ids, _ = build_datasets(df)
    print(f"  {len(station_ids)} stations in training set")

    # --- logger ---
    logger = None
    if use_wandb:
        try:
            import wandb
            from lightning.pytorch.loggers import WandbLogger
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            logger = WandbLogger(
                project=os.getenv("WANDB_PROJECT", "air-quality-forecasting"),
                name="deepar-primary",
                log_model=False,
            )
        except Exception as e:
            print(f"W&B init failed ({e}); logging to console only.")

    trainer_kwargs = {
        "max_epochs":      epochs,
        "accelerator":     "auto",
        "gradient_clip_val": 0.1,
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=True),
        ],
        "enable_progress_bar": True,
        "log_every_n_steps":   10,
    }
    if logger:
        trainer_kwargs["logger"] = logger

    print("Building DeepAR estimator...")
    estimator = build_estimator(
        cardinality=[n_stations],
        trainer_kwargs=trainer_kwargs,
    )

    print("Training...")
    predictor = estimator.train(
        training_data=train_ds,
        validation_data=val_ds,
    )

    PREDICTOR_PATH.mkdir(parents=True, exist_ok=True)
    predictor.serialize(PREDICTOR_PATH)
    print(f"\nPredictor saved to {PREDICTOR_PATH}")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepAR PM2.5 forecaster")
    parser.add_argument("--epochs",   type=int,  default=50)
    parser.add_argument("--patience", type=int,  default=10)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        patience=args.patience,
        use_wandb=not args.no_wandb,
    )
