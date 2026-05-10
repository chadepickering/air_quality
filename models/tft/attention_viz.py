"""
Step 7.5 — TFT attention and variable selection visualization.

Usage:
    python -m models.tft.attention_viz [--checkpoint PATH] [--output-dir DIR]

Produces two outputs per run:
    1. evaluation/tft_variable_importance.json
       Variable selection weights averaged across all val windows, split into
       encoder weights (how much each feature matters in the history window)
       and decoder weights (how much each known-future feature matters).

    2. evaluation/tft_attention_patterns.png
       Heatmap of mean attention scores: rows = decoder timestep (0..71),
       columns = encoder timestep (-168..0).  Shows which historical hours
       the model attends to most when generating each future hour's forecast.

These outputs are consumed by the Streamlit interface in Step 11.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.tft.model import (
    KNOWN_REALS,
    MAX_ENCODER_LENGTH,
    UNKNOWN_REALS,
    build_tft,
)
from models.tft.train import (
    CKPT_PATH,
    DATASET_PATH,
    load_and_prepare_df,
)

EVAL_DIR   = Path(__file__).parent.parent.parent / "evaluation"
VIZ_DIR    = EVAL_DIR
N_INTERP_BATCHES = 20   # number of val batches to average over for stability


def run(
    checkpoint: Path = CKPT_PATH,
    dataset_params: Path = DATASET_PATH,
    output_dir: Path = VIZ_DIR,
    n_batches: int = N_INTERP_BATCHES,
) -> None:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load model and rebuild val dataset ---
    print("Loading model and val dataset...")
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint))
    model.eval()

    df     = load_and_prepare_df(["val"])
    params = torch.load(dataset_params, weights_only=False)
    dataset = TimeSeriesDataSet.from_parameters(params, df, predict=False)
    loader  = dataset.to_dataloader(
        train=False, batch_size=64, num_workers=0
    )

    # --- collect interpretation outputs ---
    enc_var_weights_list: list[torch.Tensor] = []   # (batch, n_enc_features)
    dec_var_weights_list: list[torch.Tensor] = []   # (batch, n_dec_features)
    attn_list:            list[torch.Tensor] = []   # (batch, n_heads, dec_len, enc_len)

    print(f"Running interpret_output on {n_batches} val batches...")
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            out = model(x)
            interp = model.interpret_output(
                out, reduction="none"
            )
            # encoder variable selection: shape (batch, enc_len, n_features) → mean over time
            enc_w = interp["encoder_variables"].mean(dim=1)   # (batch, n_enc_features)
            # decoder variable selection: shape (batch, dec_len, n_dec_features) → mean over time
            dec_w = interp["decoder_variables"].mean(dim=1)   # (batch, n_dec_features)
            # attention: shape (batch, n_heads, dec_len, enc_len)
            attn  = interp["attention"]

            enc_var_weights_list.append(enc_w.cpu())
            dec_var_weights_list.append(dec_w.cpu())
            attn_list.append(attn.cpu())

    enc_weights = torch.cat(enc_var_weights_list).mean(dim=0).numpy()  # (n_enc_features,)
    dec_weights = torch.cat(dec_var_weights_list).mean(dim=0).numpy()  # (n_dec_features,)
    attn_mean   = torch.cat(attn_list).mean(dim=(0, 1)).numpy()        # (dec_len, enc_len)

    # --- variable importance JSON ---
    # pytorch-forecasting orders encoder features as: unknown_reals + known_reals + target
    enc_feature_names = UNKNOWN_REALS + KNOWN_REALS + ["pm25_target_history"]
    dec_feature_names = KNOWN_REALS

    # Clip to actual tensor width (dataset may drop or reorder some features)
    enc_feature_names = enc_feature_names[: len(enc_weights)]
    dec_feature_names = dec_feature_names[: len(dec_weights)]

    enc_importance = dict(sorted(
        zip(enc_feature_names, enc_weights.tolist()),
        key=lambda kv: -kv[1],
    ))
    dec_importance = dict(sorted(
        zip(dec_feature_names, dec_weights.tolist()),
        key=lambda kv: -kv[1],
    ))

    importance_path = output_dir / "tft_variable_importance.json"
    importance_path.write_text(json.dumps({
        "encoder_variable_importance": enc_importance,
        "decoder_variable_importance": dec_importance,
    }, indent=2))
    print(f"Variable importance saved to {importance_path}")

    print("\nTop 5 encoder features by importance:")
    for name, w in list(enc_importance.items())[:5]:
        print(f"  {name:<30s}  {w:.4f}")

    # --- attention heatmap PNG ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(
            attn_mean,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_xlabel("Encoder timestep (hours before prediction)")
        ax.set_ylabel("Decoder timestep (hours ahead)")
        ax.set_title("TFT Mean Attention Weights (val set)")

        # x-axis: label as negative hours (−168 to −1)
        n_enc = attn_mean.shape[1]
        x_ticks = list(range(0, n_enc, 24))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"−{n_enc - t}" for t in x_ticks])

        # y-axis: label as forecast hours ahead (1 to 72)
        n_dec = attn_mean.shape[0]
        y_ticks = list(range(0, n_dec, 12))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(t + 1) for t in y_ticks])

        plt.colorbar(im, ax=ax, label="Attention weight")
        plt.tight_layout()

        attn_path = output_dir / "tft_attention_patterns.png"
        fig.savefig(attn_path, dpi=150)
        plt.close(fig)
        print(f"Attention heatmap saved to {attn_path}")

    except ImportError:
        print("matplotlib not installed — skipping attention heatmap PNG.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFT attention and variable importance")
    parser.add_argument("--checkpoint",     type=Path, default=CKPT_PATH)
    parser.add_argument("--dataset-params", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir",     type=Path, default=VIZ_DIR)
    parser.add_argument("--n-batches",      type=int,  default=N_INTERP_BATCHES)
    args = parser.parse_args()

    run(
        checkpoint=args.checkpoint,
        dataset_params=args.dataset_params,
        output_dir=args.output_dir,
        n_batches=args.n_batches,
    )
