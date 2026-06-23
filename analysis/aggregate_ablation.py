"""
aggregate_ablation.py  —  summarize the Week-2 fixed-schedule ablation grid

Reads all results/raw_ablation/*.json (200 files: rate x eps x dataset x seed),
groups by (dataset, rate, eps), averages ordering across seeds, and derives
the epoch at which each fixed schedule reached dim=128.

The ablation JSONs do NOT store a reach-128 epoch, but they store
n_growth_events, and the ProgrammaticTrigger grows on a fixed clock
(one growth every `epochs_per_stage` epochs). So:
    epoch_reached_128 = n_growth_events * epochs_per_stage
when the schedule reached 128 within the budget. This was verified against
a known cell (rate1.3/eps4 -> 12 events -> epoch 48).

Writes results/processed/ablation_summary.json, which plot_frontier.py can
read so the fixed-schedule anchor numbers come straight from data.

Usage:
    python aggregate_ablation.py
    python aggregate_ablation.py --dataset cifar10            # print one dataset
    python aggregate_ablation.py --total-epochs 60
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path("results/raw_ablation")
OUT = Path("results/processed/ablation_summary.json")
START_DIM, MAX_DIM = 6, 128


def derive_epoch_128(rate, eps, n_growth_events, total_epochs):
    """
    Reach-128 epoch from the deterministic growth clock. The last growth
    lands on 128, at epoch n_growth_events * eps. If that exceeds the budget
    (force-completed schedules), the schedule effectively reaches 128 only at
    the final epoch; return total_epochs and flag it.
    """
    derived = n_growth_events * eps
    if derived <= total_epochs:
        return derived, False        # reached on the natural clock
    return total_epochs, True        # force-completed at end of budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="filter to one dataset for printing")
    ap.add_argument("--total-epochs", type=int, default=60)
    args = ap.parse_args()

    files = glob.glob(str(RAW / "*.json"))
    if not files:
        print(f"No JSONs found in {RAW}/")
        return

    groups = defaultdict(list)
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        key = (r["dataset"], float(r["rate"]), int(r["epochs_per_stage"]))
        groups[key].append(r)

    summary = []
    for (ds, rate, eps), rs in sorted(groups.items()):
        ordv = np.array([r["ordering_variance_decay"] for r in rs])
        lp = np.array([r["linear_probe"] for r in rs])
        nge = int(round(np.median([r["n_growth_events"] for r in rs])))
        reached_flags = [bool(r.get("reached_naturally", True)) for r in rs]
        e128, forced = derive_epoch_128(rate, eps, nge, args.total_epochs)
        summary.append({
            "dataset": ds, "rate": rate, "epochs_per_stage": eps,
            "config": f"rate{rate}/eps{eps}",
            "n_seeds": len(rs),
            "ordering_mean": float(ordv.mean()),
            "ordering_std": float(ordv.std()),
            "linear_probe_mean": float(lp.mean()),
            "n_growth_events": nge,
            "epoch_reached_128": e128,
            "force_completed": forced,
            "all_reached_naturally": all(reached_flags),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    # print table
    rows = [s for s in summary if (args.dataset is None or s["dataset"] == args.dataset)]
    print(f"\n{'dataset':<10}{'config':<16}{'n':>3}{'ordering':>16}"
          f"{'LP':>9}{'epoch128':>10}{'forced':>8}")
    print("-" * 72)
    for s in sorted(rows, key=lambda d: (d["dataset"], d["rate"], d["epochs_per_stage"])):
        print(f"{s['dataset']:<10}{s['config']:<16}{s['n_seeds']:>3}"
              f"{s['ordering_mean']:>9.3f}+/-{s['ordering_std']:.3f}"
              f"{s['linear_probe_mean']:>9.4f}{s['epoch_reached_128']:>10}"
              f"{'yes' if s['force_completed'] else '':>8}")
    print("-" * 72)
    print(f"wrote {OUT}  ({len(summary)} configs)")
    print("\nFor the frontier figure, pick ~3-4 anchors spanning the tradeoff,")
    print("e.g. a strongly-ordered slow one, the rate1.7/eps8 sweet spot, and")
    print("a disordered fast one (low ordering) for the inset.")


if __name__ == "__main__":
    main()
