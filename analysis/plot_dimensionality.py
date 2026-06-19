"""
plot_dimensionality.py
Replicates the paper's Fig 2B (intrinsic dimensionality over epochs).

Run from project root:
    python analysis/plot_dimensionality.py --dataset cifar10

Reads:  results/raw/{dataset}_baseline_{ae,devae_fixed,pca_ae}_seed*.json
        (requires per-epoch "intrinsic_dim" in each history entry)
Writes: results/figures/{dataset}_dimensionality.png  (and .pdf)
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt


MODEL_CONFIG = {
    "baseline_ae":         {"label": "AE",     "color": "#1a7adb"},
    "baseline_pca_ae":     {"label": "PCA-AE", "color": "#2ca02c"},
    "baseline_devae_fixed":{"label": "Dev-AE", "color": "#e82817"},
}


def load_id_curves(dataset, experiment):
    """Load per-epoch intrinsic_dim curves across all seeds."""
    files = sorted(glob.glob(f"results/raw/{dataset}_{experiment}_seed*.json"))
    if not files:
        print(f"  [warn] no files for {dataset}_{experiment}")
        return None

    curves = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        hist = data["history"]
        if "intrinsic_dim" not in hist[0]:
            print(f"  [warn] {fp} has no per-epoch intrinsic_dim — rerun with ID logging")
            return None
        curves.append([h["intrinsic_dim"] for h in hist])

    min_len = min(len(c) for c in curves)
    return np.array([c[:min_len] for c in curves])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    args = parser.parse_args()
    ds = args.dataset

    os.makedirs("results/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    for experiment, cfg in MODEL_CONFIG.items():
        curves = load_id_curves(ds, experiment)
        if curves is None:
            continue
        epochs = np.arange(1, curves.shape[1] + 1)
        mean, std = np.nanmean(curves, 0), np.nanstd(curves, 0)
        ax.plot(epochs, mean, color=cfg["color"], linewidth=2, label=cfg["label"])
        ax.fill_between(epochs, mean - std, mean + std, color=cfg["color"], alpha=0.18)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Intrinsic Dimensionality")
    ax.set_title(f"Latent dimensionality over training ({ds.upper()})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    png = f"results/figures/{ds}_dimensionality.png"
    pdf = f"results/figures/{ds}_dimensionality.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"Saved {png} and {pdf}")


if __name__ == "__main__":
    main()
