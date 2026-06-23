"""
plot_decoupling.py  —  ordering-vs-accuracy decoupling scatter (paper figure)

The central finding in one panel: across the full fixed-schedule grid, ordering
(variance-decay rho) swings across nearly its entire range while linear-probe
accuracy stays almost flat. Ordering and accuracy are decoupled.

Each point is one (rate, eps) config (mean over seeds), x = linear probe,
y = ordering. A near-vertical cloud = accuracy barely moves while ordering
spans the range. Points are colored by growth rate to show the phase boundary.

Data source: results/processed/ablation_summary.json (from aggregate_ablation.py).

Usage:
    python plot_decoupling.py --dataset cifar10
    python plot_decoupling.py --dataset cifar100 --out results/figures/decoupling_cifar100.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ABLATION_SUMMARY = Path("results/processed/ablation_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not ABLATION_SUMMARY.exists():
        print(f"{ABLATION_SUMMARY} not found - run aggregate_ablation.py first.")
        return
    with open(ABLATION_SUMMARY) as f:
        rows = [r for r in json.load(f) if r["dataset"] == args.dataset]
    if not rows:
        print(f"No rows for {args.dataset}.")
        return

    lp = np.array([r["linear_probe_mean"] for r in rows])
    ordv = np.array([r["ordering_mean"] for r in rows])
    rates = np.array([r["rate"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    uniq = sorted(set(rates))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(uniq)))
    rate_color = {r: cmap[i] for i, r in enumerate(uniq)}
    for r in uniq:
        m = rates == r
        ax.scatter(lp[m], ordv[m], s=70, color=rate_color[r],
                   edgecolors="white", linewidths=0.6, label=f"rate {r}", zorder=3)

    # span annotations
    ax.axhline(0.0, color="#cccccc", lw=0.8, zorder=1)
    lp_span = lp.max() - lp.min()
    ord_span = ordv.max() - ordv.min()
    ax.annotate(f"accuracy span: {lp_span*100:.1f} pts\nordering span: {ord_span:.2f}",
                xy=(0.03, 0.03), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))

    ax.set_xlabel("Linear-probe accuracy", fontsize=10.5)
    ax.set_ylabel("Ordering (variance-decay \u03c1)", fontsize=10.5)
    ax.set_title(f"Ordering and accuracy are decoupled  ({args.dataset})", fontsize=12)
    ax.grid(True, alpha=0.22)
    ax.legend(title="growth rate", fontsize=8.5, title_fontsize=9, loc="center right")

    # pad x so the near-vertical cloud is readable
    xmid = lp.mean()
    half = max(lp_span, 0.02) * 2.5
    ax.set_xlim(xmid - half, xmid + half)

    fig.tight_layout()
    out = args.out or f"results/figures/decoupling_{args.dataset}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
