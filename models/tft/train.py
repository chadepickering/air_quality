"""
Step 7.2 / 7.3 — TFT data preparation and training script.

Usage:
    python -m models.tft.train [options]

    --epochs      Max training epochs (default: 50)
    --batch-size  Mini-batch size     (default: 128)
    --lr          Initial learning rate (default: 1e-3)
    --patience    Early-stopping patience in epochs (default: 5)
    --no-wandb    Disable W&B logging

Outputs (written to models/tft/):
    best_model.ckpt   — Lightning checkpoint with lowest val_loss
    dataset_params.pt — TimeSeriesDataSet parameters for loading at inference

Key design decisions vs LSTM:
  - Encoder: 168hr (7-day) lookback vs LSTM's 24hr — TFT attention can directly
    reference same-hour-last-week patterns.
  - Decoder: 72hr full output; HORIZONS (3, 12, 24, 72) extracted by index.
  - Known-future reals: calendar features (hour, day, month, is_weekend) only —
    these are genuinely knowable in advance.
  - Unknown reals: all pollutant readings, rolling/lag/spatial features.
  - GroupNormalizer: per-station z-score, fit on train split only.
  - Quantile output [0.05, 0.50, 0.95]: p50 is point forecast; p5/p95 form 90% PI.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

from ingestion.database import DB_PATH
from models.tft.model import (
    KNOWN_REALS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    STATIC_CATS,
    TARGET,
    UNKNOWN_REALS,
    build_tft,
)
from streaming.feature_engineering import TRAIN_END

MODEL_DIR    = Path(__file__).parent
CKPT_PATH    = MODEL_DIR / "best_model.ckpt"
DATASET_PATH = MODEL_DIR / "dataset_params.pt"

ALL_FEATURE_COLS = KNOWN_REALS + UNKNOWN_REALS + [TARGET]


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_and_prepare_df(splits: list[str]) -> pd.DataFrame:
    """
    Load processed_features for the given splits, add time_idx, fill NaN.

    time_idx is computed as hours since the global minimum timestamp across all
    stations and all requested splits.  This makes time_idx comparable across
    groups without requiring all stations to start at the same hour.

    NaN-fill strategy (within each station):
        1. Forward-fill — handles lag/roll NaN that arise mid-series.
        2. Backward-fill — handles NaN at the start of the record (lag1/3/24
           have no valid predecessor for the first rows).
        3. Fill remaining with 0 — fallback for fully-null columns.
    """
    split_list = ", ".join(f"'{s}'" for s in splits)
    cols = ["station_id", "timestamp"] + ALL_FEATURE_COLS + ["split"]

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT {', '.join(cols)}
        FROM processed_features
        WHERE split IN ({split_list})
        ORDER BY station_id, timestamp
    """).df()
    con.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # is_weekend is boolean in DuckDB — cast to float for pytorch-forecasting
    df["is_weekend"] = df["is_weekend"].astype(float)

    # time_idx: hours since global minimum timestamp
    global_min = df["timestamp"].min()
    df["time_idx"] = ((df["timestamp"] - global_min) / pd.Timedelta(hours=1)).astype(int)

    # NaN fill within each station
    feature_cols = KNOWN_REALS + UNKNOWN_REALS + [TARGET]
    df[feature_cols] = (
        df.groupby("station_id", group_keys=False)[feature_cols]
        .apply(lambda g: g.ffill().bfill().fillna(0.0))
    )

    return df


def build_datasets(df: pd.DataFrame, batch_size: int = 128):
    """
    Build TimeSeriesDataSet for train and val, and return their DataLoaders.

    The val dataset is built via from_dataset() so it inherits the GroupNormalizer
    fit on training data only — no leakage.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    train_cutoff = int(
        (pd.Timestamp(str(TRAIN_END), tz=None) - df["timestamp"].min())
        / pd.Timedelta(hours=1)
    )

    train_df = df[df["time_idx"] <= train_cutoff].copy()
    val_df   = df.copy()   # val dataset needs encoder context from train period

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=STATIC_CATS,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=STATIC_CATS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=STATIC_CATS, transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    # Save dataset params so evaluate.py can reconstruct without refitting
    torch.save(training.get_parameters(), DATASET_PATH)

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=False,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )

    return training, validation, train_loader, val_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs:     int   = 50,
    batch_size: int   = 128,
    lr:         float = 1e-3,
    patience:   int   = 5,
    use_wandb:  bool  = True,
) -> None:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    print("Loading and preparing data...")
    df = load_and_prepare_df(["train", "val"])
    print(f"  {len(df):,} rows, {df['station_id'].nunique()} stations")

    print("Building TimeSeriesDataSets...")
    training, _, train_loader, val_loader = build_datasets(df, batch_size=batch_size)
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    print("Building TFT model...")
    model = build_tft(training)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- callbacks ---
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=True),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(MODEL_DIR),
            filename="best_model",
            save_top_k=1,
            mode="min",
        ),
    ]

    # --- logger ---
    logger = None
    if use_wandb:
        try:
            import wandb
            from lightning.pytorch.loggers import WandbLogger
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            logger = WandbLogger(
                project=os.getenv("WANDB_PROJECT", "air-quality-forecasting"),
                name="tft-baseline",
                log_model=False,
            )
        except Exception as e:
            print(f"W&B init failed ({e}); logging to console only.")
            logger = None

    # --- trainer ---
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    print("Training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best = callbacks[1].best_model_path
    print(f"\nBest checkpoint: {best}")
    if logger and hasattr(logger, "experiment"):
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFT PM2.5 forecaster")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=128)
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
