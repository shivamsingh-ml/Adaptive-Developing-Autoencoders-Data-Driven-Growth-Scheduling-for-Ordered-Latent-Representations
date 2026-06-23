"""
plot_trigger_signal.py  —  trigger-signal curve (paper Fig 4)

Shows the per-epoch intrinsic-dimensionality trace the ID-trigger reads,
with growth events marked as vertical lines and the bottleneck dim shown
as a step line on a secondary axis. This visualizes the trigger detecting
per-stage convergence and growing in response.

Data source: a single representative run's per-epoch log. By default it
reads results/processed/trigger_smoketest.json (which stores the ID signal
per epoch + the growth schedule). You can also point it at any run JSON
that contains an "id_curve" (or "signal") list and a "growth_schedule"
list of (epoch, old_dim, new_dim) tuples.

Usage:
    python plot_trigger_signal.py
    python plot_trigger_signal.py --infile results/processed/trigger_smoketest.json
    python plot_trigger_signal.py --out results/figures/trigger_signal.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ADAPTIVE_C = "#2b6cb0"
GROWTH_C = "#c0392b"
DIM_C = "#7a7a7a"


def load_run(infile):
    """
    Return (id_curve, growth_events) from a run JSON.
    Accepts either the smoketest format (list of result dicts) or a single
    result dict. id_curve from "id_curve" or "signal"; growth events from
    "growth_schedule" or "events" as (epoch, old, new) triples.
    """
    with open(infile) as f:
        data = json.load(f)

    # smoketest stores a list; pick the ID-trigger entry
    if isinstance(data, list):
        rec = next((r for r in data if r.get("name", "").lower().startswith("id")), data[0])
    else:
        rec = data

    id_curve = rec.get("id_curve") or rec.get("signal")
    events = rec.get("growth_schedule") or rec.get("events") or []
    # normalize events to (epoch, old, new)
    norm = []
    for e in events:
        if isinstance(e, (list, tuple)) and len(e) >= 3:
            norm.append((int(e[0]), int(e[1]), int(e[2])))
    return id_curve, norm


def build_dim_steps(n_epochs, events, start_dim=6):
    dims = [start_dim] * n_epochs
    cur = start_dim
    ev = {ep: new for ep, old, new in events}
    for e in range(n_epochs):
        if e in ev:
            cur = ev[e]
        dims[e] = cur
    return dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/processed/trigger_smoketest.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--start-dim", type=int, default=6)
    args = ap.parse_args()

    id_curve, events = load_run(args.infile)
    if not id_curve:
        print(f"No id_curve/signal found in {args.infile}. "
              f"Point --infile at a run JSON that logs the per-epoch ID.")
        return

    id_curve = [v if v is not None else np.nan for v in id_curve]
    n = len(id_curve)
    epochs = np.arange(n)
    dims = build_dim_steps(n, events, start_dim=args.start_dim)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # ID signal
    ax.plot(epochs, id_curve, "-o", color=ADAPTIVE_C, markersize=3.5,
            linewidth=1.6, label="Intrinsic dimensionality (TwoNN)")

    # growth event markers
    for ep, old, new in events:
        ax.axvline(ep, color=GROWTH_C, ls="--", lw=1.0, alpha=0.7, zorder=1)
        ymax = np.nanmax(id_curve)
        ax.annotate(f"{old}\u2192{new}", (ep, ymax),
                    textcoords="offset points", xytext=(2, -2),
                    fontsize=7.5, color=GROWTH_C, rotation=90, va="top")

    ax.set_xlabel("Epoch", fontsize=10.5)
    ax.set_ylabel("Intrinsic dimensionality", fontsize=10.5, color=ADAPTIVE_C)
    ax.tick_params(axis="y", labelcolor=ADAPTIVE_C)
    ax.grid(True, alpha=0.22)

    # secondary axis: bottleneck dim as a step line
    ax2 = ax.twinx()
    ax2.step(epochs, dims, where="post", color=DIM_C, lw=1.3, alpha=0.8,
             label="Bottleneck dim")
    ax2.set_ylabel("Bottleneck dimension", fontsize=10.5, color=DIM_C)
    ax2.tick_params(axis="y", labelcolor=DIM_C)
    ax2.set_ylim(0, max(dims) * 1.1)

    # combined legend
    h1 = [plt.Line2D([0], [0], color=ADAPTIVE_C, marker="o", lw=1.6, label="Intrinsic dim (TwoNN)"),
          plt.Line2D([0], [0], color=DIM_C, lw=1.3, label="Bottleneck dim"),
          plt.Line2D([0], [0], color=GROWTH_C, ls="--", lw=1.0, label="Growth event")]
    ax.legend(handles=h1, loc="lower right", fontsize=8.5, framealpha=0.95)

    ax.set_title("ID-trigger: convergence signal and growth events", fontsize=12)
    fig.tight_layout()

    out = args.out or "results/figures/trigger_signal.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
