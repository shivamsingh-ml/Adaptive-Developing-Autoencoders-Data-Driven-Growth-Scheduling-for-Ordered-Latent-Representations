"""
plot_frontier.py  —  ordering-vs-efficiency frontier figure (full-grid cloud)

Scatters ALL fixed-schedule configs from the Week-2 ablation as a cloud in
(epoch-reached-128, ordering) space, and overlays the adaptive ID-trigger
operating points (fast/mid/slow) as a line with std bars.

The visual claim: the fixed-schedule family occupies a broad region — you
must SEARCH it to find efficient, ordered configs — while the adaptive
trigger lands on the upper edge automatically, without the search.

Data sources (both produced by the pipeline, no hand-typed numbers):
  - fixed cloud: results/processed/ablation_summary.json  (aggregate_ablation.py)
  - adaptive:    results/raw_frontier/{dataset}_{point}_seed*.json (run_frontier.py)

Strongly disordered fixed configs (ordering < ORDERED_CUT) would force the
y-axis to span the full [-0.7, 1.0] and flatten the frontier. The main panel
zooms to the ordered band; a small inset shows the full range for context.

Usage:
    python plot_frontier.py --dataset cifar10
    python plot_frontier.py --dataset cifar10 --out results/figures/frontier_cifar10.png
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RAW_FRONTIER = Path("results/raw_frontier")
ABLATION_SUMMARY = Path("results/processed/ablation_summary.json")

POINT_ORDER = {"fast": 0, "mid": 1, "slow": 2}
POINT_LABEL = {"fast": "Fast (p=3)", "mid": "Mid (p=4)", "slow": "Slow (p=5)"}
ADAPTIVE_LABEL_OFFSET = {"fast": (-4, 12), "mid": (4, -20), "slow": (-8, 12)}

ORDERED_CUT = 0.4            # fixed points below this -> inset only
ADAPTIVE_C = "#2b6cb0"
FIXED_C    = "#9a9a9a"
EDGE_C     = "#c0392b"


def load_adaptive(dataset):
    by_point = {}
    for fp in glob.glob(str(RAW_FRONTIER / f"{dataset}_*_seed*.json")):
        with open(fp) as f:
            r = json.load(f)
        by_point.setdefault(r["point"], []).append(r)
    points = []
    for pt, rs in by_point.items():
        reached = [r for r in rs if r.get("reached_128")]
        if not reached:
            print(f"WARNING: adaptive point '{pt}' never reached 128 - skipping")
            continue
        if len(reached) < len(rs):
            print(f"NOTE: adaptive '{pt}': {len(reached)}/{len(rs)} seeds reached 128")
        ordv = np.array([r["ordering_variance_decay"] for r in reached])
        e128 = np.array([r["epoch_reached_128"] for r in reached])
        points.append(dict(point=pt,
                           ord_mean=float(ordv.mean()), ord_std=float(ordv.std()),
                           e128_mean=float(e128.mean()), e128_std=float(e128.std()),
                           n=len(reached)))
    points.sort(key=lambda d: POINT_ORDER.get(d["point"], 9))
    return points


def load_fixed(dataset):
    if not ABLATION_SUMMARY.exists():
        print(f"WARNING: {ABLATION_SUMMARY} not found - run aggregate_ablation.py first. "
              f"No fixed cloud will be drawn.")
        return []
    with open(ABLATION_SUMMARY) as f:
        allrows = json.load(f)
    return [r for r in allrows if r["dataset"] == dataset]


def upper_edge(fixed):
    """Upper-left envelope: best ordering reachable by epoch x, as x increases."""
    pts = sorted(fixed, key=lambda d: d["epoch_reached_128"])
    edge = []
    best = -np.inf
    for d in pts:
        if d["ordering_mean"] > best:
            best = d["ordering_mean"]
            edge.append((d["epoch_reached_128"], best))
    return edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-edge", action="store_true", help="hide the fixed upper-edge line")
    args = ap.parse_args()

    adaptive = load_adaptive(args.dataset)
    fixed = load_fixed(args.dataset)
    if not adaptive and not fixed:
        print("Nothing to plot.")
        return

    ordered_fixed = [d for d in fixed if d["ordering_mean"] >= ORDERED_CUT]
    disordered_fixed = [d for d in fixed if d["ordering_mean"] < ORDERED_CUT]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # fixed cloud (ordered region)
    if ordered_fixed:
        ax.scatter([d["epoch_reached_128"] for d in ordered_fixed],
                   [d["ordering_mean"] for d in ordered_fixed],
                   marker="s", s=46, c=FIXED_C, alpha=0.75,
                   edgecolors="white", linewidths=0.5, zorder=3)

    # fixed upper edge
    if ordered_fixed and not args.no_edge:
        edge = upper_edge(ordered_fixed)
        if len(edge) >= 2:
            ex, ey = zip(*edge)
            ax.plot(ex, ey, "--", color=EDGE_C, lw=1.3, alpha=0.8, zorder=4)

    # adaptive frontier line
    if adaptive:
        ax_x = [d["e128_mean"] for d in adaptive]
        ax_y = [d["ord_mean"] for d in adaptive]
        xerr = [d["e128_std"] for d in adaptive]
        yerr = [d["ord_std"] for d in adaptive]
        ax.errorbar(ax_x, ax_y, xerr=xerr, yerr=yerr, fmt="-o",
                    color=ADAPTIVE_C, ecolor=ADAPTIVE_C, elinewidth=1.3,
                    capsize=3, markersize=9, linewidth=1.9,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        for d in adaptive:
            off = ADAPTIVE_LABEL_OFFSET.get(d["point"], (8, 8))
            ax.annotate(POINT_LABEL.get(d["point"], d["point"]),
                        (d["e128_mean"], d["ord_mean"]),
                        textcoords="offset points", xytext=off,
                        fontsize=9.5, color=ADAPTIVE_C, fontweight="bold", zorder=7)

    # zoom to ordered band
    ys = ([d["ord_mean"] for d in adaptive] +
          [d["ordering_mean"] for d in ordered_fixed])
    yerrs = ([d["ord_std"] for d in adaptive] + [0.0] * len(ordered_fixed))
    xs = ([d["e128_mean"] for d in adaptive] +
          [d["epoch_reached_128"] for d in ordered_fixed])
    if ys:
        ax.set_ylim(min(y - e for y, e in zip(ys, yerrs)) - 0.04, max(ys) + 0.05)
    if xs:
        ax.set_xlim(min(xs) - 4, max(xs) + 6)

    ax.set_xlabel("Epoch at which dim=128 is reached   (faster \u2190)", fontsize=10.5)
    ax.set_ylabel("Ordering (variance-decay \u03c1)", fontsize=10.5)
    ax.set_title(f"Ordering\u2013efficiency frontier  ({args.dataset})", fontsize=12)
    ax.grid(True, alpha=0.22)

    # custom legend
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=FIXED_C,
               markeredgecolor="white", markersize=8,
               label="Fixed schedules (20-cell grid)"),
        Line2D([0], [0], color=ADAPTIVE_C, marker="o", markersize=8,
               markeredgecolor="white", lw=1.9, label="ID-trigger (adaptive)"),
    ]
    if not args.no_edge:
        handles.insert(1, Line2D([0], [0], color=EDGE_C, ls="--", lw=1.3,
                                 label="Fixed upper edge"))
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)

    # inset: full range incl. disordered fixed configs
    if disordered_fixed:
        axin = ax.inset_axes([0.06, 0.50, 0.36, 0.42])
        axin.scatter([d["epoch_reached_128"] for d in fixed],
                     [d["ordering_mean"] for d in fixed],
                     marker="s", s=22, c=FIXED_C, alpha=0.8, zorder=3)
        if adaptive:
            axin.plot([d["e128_mean"] for d in adaptive],
                      [d["ord_mean"] for d in adaptive],
                      "-o", color=ADAPTIVE_C, markersize=3.5, lw=1.1, zorder=4)
        axin.axhline(0.0, color="#cccccc", lw=0.7)
        axin.set_ylim(-0.8, 1.0)
        axin.set_title("full range (incl. disordered)", fontsize=7.5)
        axin.tick_params(labelsize=7)

    fig.tight_layout()
    out = args.out or f"results/figures/frontier_{args.dataset}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()