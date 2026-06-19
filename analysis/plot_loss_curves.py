"""
plot_loss_curves.py
Replicates the paper's Fig 2A loss-curve panel from saved run JSONs.

Place this in analysis/ and run from the project root:
    python analysis/plot_loss_curves.py --dataset cifar10

Reads:  results/raw/{dataset}_baseline_{ae,devae_fixed,pca_ae}_seed*.json
Writes: results/figures/{dataset}_loss_curves.png  (and .pdf)
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


def load_curves(dataset, experiment):
    """Load train/val loss curves across all seeds for one experiment."""
    pattern = f"results/raw/{dataset}_{experiment}_seed*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  [warn] no files for {pattern}")
        return None, None, None

    train_curves, val_curves, growth_epochs = [], [], None
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        hist = data["history"]
        train_curves.append([h["train_loss"] for h in hist])
        val_curves.append([h["val_loss"] for h in hist])
        if growth_epochs is None and data.get("growth_epochs"):
            growth_epochs = data["growth_epochs"]

    # Trim to common length (in case of ragged runs)
    min_len = min(len(c) for c in train_curves)
    train = np.array([c[:min_len] for c in train_curves])
    val = np.array([c[:min_len] for c in val_curves])
    return train, val, growth_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    args = parser.parse_args()
    ds = args.dataset

    os.makedirs("results/figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    dev_growth = None
    for experiment, cfg in MODEL_CONFIG.items():
        train, val, growth_epochs = load_curves(ds, experiment)
        if train is None:
            continue

        epochs = np.arange(1, train.shape[1] + 1)
        train_mean, train_std = train.mean(0), train.std(0)
        val_mean = val.mean(0)

        # Solid = train (mean + shaded std), dashed = val mean
        ax.plot(epochs, train_mean, color=cfg["color"], linewidth=2,
                label=f"{cfg['label']} (train)")
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                        color=cfg["color"], alpha=0.15)
        ax.plot(epochs, val_mean, color=cfg["color"], linewidth=1.2,
                linestyle="--", alpha=0.8, label=f"{cfg['label']} (val)")

        if experiment == "baseline_devae_fixed":
            dev_growth = growth_epochs

    # Mark Dev-AE growth events with faint vertical lines
    if dev_growth:
        for ge in dev_growth:
            ax.axvline(ge + 1, color="gray", linestyle=":", linewidth=0.7, alpha=0.4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Reconstruction loss over training ({ds.upper()})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    plt.tight_layout()

    png = f"results/figures/{ds}_loss_curves.png"
    pdf = f"results/figures/{ds}_loss_curves.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"Saved {png} and {pdf}")


if __name__ == "__main__":
    main()
