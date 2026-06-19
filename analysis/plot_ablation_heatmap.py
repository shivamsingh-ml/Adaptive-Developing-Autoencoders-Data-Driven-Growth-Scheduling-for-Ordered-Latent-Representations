"""
plot_ablation_heatmap.py  —  Week-2 phase-boundary figure

Reads results/raw_ablation/*.json and produces heatmaps of ordering
(and linear probe) as a function of growth_rate x epochs_per_stage,
averaged across seeds. Shows the ordered/disordered phase boundary.

Usage:
    python analysis/plot_ablation_heatmap.py --dataset cifar10
    python analysis/plot_ablation_heatmap.py --dataset cifar100
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


RATES = [1.3, 1.5, 1.7, 2.0, 2.5]
EPOCHS_PER_STAGE = [4, 6, 8, 10]


def load_grid(dataset, metric):
    """Return a [len(EPOCHS) x len(RATES)] matrix of mean metric values."""
    buckets = defaultdict(list)
    for fp in glob.glob(f"results/raw_ablation/{dataset}_rate*_eps*_seed*.json"):
        with open(fp) as f:
            r = json.load(f)
        buckets[(r["rate"], r["epochs_per_stage"])].append(r[metric])

    grid = np.full((len(EPOCHS_PER_STAGE), len(RATES)), np.nan)
    counts = np.zeros_like(grid)
    for i, eps in enumerate(EPOCHS_PER_STAGE):
        for j, rate in enumerate(RATES):
            vals = buckets.get((rate, eps), [])
            if vals:
                grid[i, j] = np.mean(vals)
                counts[i, j] = len(vals)
    return grid, counts


def plot_heatmap(ax, grid, title, cmap, vmin, vmax, fmt="{:.2f}"):
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(RATES)))
    ax.set_xticklabels(RATES)
    ax.set_yticks(range(len(EPOCHS_PER_STAGE)))
    ax.set_yticklabels(EPOCHS_PER_STAGE)
    ax.set_xlabel("Growth rate")
    ax.set_ylabel("Epochs per stage")
    ax.set_title(title)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                # contrast-aware text color
                norm = (grid[i, j] - vmin) / (vmax - vmin + 1e-9)
                color = "white" if norm < 0.4 or norm > 0.8 else "black"
                ax.text(j, i, fmt.format(grid[i, j]),
                        ha="center", va="center", color=color, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    args = ap.parse_args()
    ds = args.dataset

    os.makedirs("results/figures", exist_ok=True)

    ord_grid, ord_counts = load_grid(ds, "ordering_variance_decay")
    lp_grid, _ = load_grid(ds, "linear_probe")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    im0 = plot_heatmap(axes[0], ord_grid,
                       f"Ordering (variance-decay) — {ds.upper()}",
                       cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="ordering rho")

    im1 = plot_heatmap(axes[1], lp_grid,
                       f"Linear probe accuracy — {ds.upper()}",
                       cmap="viridis", vmin=np.nanmin(lp_grid),
                       vmax=np.nanmax(lp_grid), fmt="{:.3f}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="accuracy")

    plt.tight_layout()
    png = f"results/figures/{ds}_ablation_heatmap.png"
    pdf = f"results/figures/{ds}_ablation_heatmap.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"Saved {png} and {pdf}")

    # Coverage report
    missing = int((ord_counts == 0).sum())
    if missing:
        print(f"[warn] {missing} grid cells have no runs yet")
    else:
        print(f"All {ord_counts.size} cells populated "
              f"(mean {ord_counts.mean():.0f} seeds/cell)")


if __name__ == "__main__":
    main()
