"""
run_frontier.py  —  Week-4 ordering-vs-efficiency frontier for the ID-trigger

Runs the ID-trigger at three operating points that trade growth speed for
ordering, each reaching dim=128 (valid operating points), so they can be
plotted as a frontier and overlaid on the Week-2 fixed-schedule grid.

Operating points (patience and max_epochs raised together so the slow
config still completes within the 60-epoch budget):
    FAST:  patience=3, max_epochs=9
    MID:   patience=4, max_epochs=11
    SLOW:  patience=5, max_epochs=13
All use min_epochs_per_stage=3, smooth_window=3, delta_threshold=0.3.

Resumable: skips any (point, dataset, seed) whose JSON already exists.

Usage:
    python run_frontier.py --dataset cifar10
    python run_frontier.py --dataset cifar10 --point fast --seed 0   # single cell
    python run_frontier.py --summary
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import build_dataloaders
from models.conv_autoencoder import ConvAutoencoder
from solver import train_with_trigger
from triggers.id_trigger import IDTrigger
from evaluation.linear_probe import linear_probe_eval
from evaluation.knn_eval import knn_eval
from evaluation.ordering_score import compute_ordering_scores


TOTAL_EPOCHS = 60
START_DIM, MAX_DIM = 6, 128
SEEDS = [0, 1, 2, 3, 4]
OUT = Path("results/raw_frontier")

# Three valid operating points along the speed/ordering frontier.
POINTS = {
    "fast": dict(patience=3, min_epochs_per_stage=3, max_epochs_per_stage=9,
                 smooth_window=3, delta_threshold=0.3),
    "mid":  dict(patience=4, min_epochs_per_stage=3, max_epochs_per_stage=11,
                 smooth_window=3, delta_threshold=0.3),
    "slow": dict(patience=5, min_epochs_per_stage=3, max_epochs_per_stage=11,
                 smooth_window=3, delta_threshold=0.3),
}

# Fixed-schedule frontier points from the Week-2 grid (for overlay reference).
# (ordering, epoch_reached_128) per config; fill in from your grid as needed.
FIXED_GRID = {
    "cifar10": [
        {"config": "rate2.0/eps4",  "ordering": -0.68, "epoch_128": 24},
        {"config": "rate1.7/eps8",  "ordering": 0.88,  "epoch_128": 48},
        {"config": "rate1.7/eps10", "ordering": 0.90,  "epoch_128": 58},
    ],
    "cifar100": [
        {"config": "rate1.7/eps8",  "ordering": 0.90,  "epoch_128": 48},
    ],
}


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def base_cfg(dataset, seed):
    return {
        "experiment": {"name": "frontier", "seed": seed, "device": "cuda"},
        "data": {"dataset": dataset, "data_dir": "./data",
                 "batch_size": 128, "num_workers": 0},
        "model": {"type": "conv_autoencoder", "start_dim": START_DIM, "max_dim": MAX_DIM},
        "training": {"epochs": TOTAL_EPOCHS, "lr": 0.1, "optimizer": "sgd", "loss": "mse"},
    }


def make_trigger(point):
    return IDTrigger(START_DIM, MAX_DIM, growth_rate=1.7, **POINTS[point])


@torch.no_grad()
def get_enc_inputs(model, loader, device, max_samples=2000):
    model.eval()
    embs, flats = [], []
    n = 0
    for x, _ in loader:
        embs.append(model.encode(x.to(device)).detach().cpu())
        flats.append(x.view(x.size(0), -1))
        n += x.size(0)
        if n >= max_samples:
            break
    return (torch.cat(embs)[:max_samples].numpy(),
            torch.cat(flats)[:max_samples].numpy())


def growth_events(history):
    return [(history[i]["epoch"], history[i-1]["latent_dim"], history[i]["latent_dim"])
            for i in range(1, len(history))
            if history[i]["latent_dim"] != history[i-1]["latent_dim"]]


def run_cell(point, dataset, seed, device, loaders=None):
    cfg = base_cfg(dataset, seed)
    if loaders is None:
        loaders = build_dataloaders(cfg)
    train_loader, val_loader, test_loader = loaders

    set_seed(seed)
    trig = make_trigger(point)
    model = ConvAutoencoder(latent_dim=trig.current_dim).to(device)

    print(f"\n{'='*70}\nFRONTIER {point.upper()} | {dataset} | seed {seed}\n{'='*70}")
    model, history = train_with_trigger(
        model=model, trigger=trig, model_cls=ConvAutoencoder,
        train_loader=train_loader, val_loader=val_loader,
        optimizer_cls=optim.SGD, loss_fn=nn.MSELoss(),
        cfg=cfg, device=device,
    )

    events = growth_events(history)
    epoch_128 = next((ep for ep, o, n in events if n == 128), None)
    lp = float(linear_probe_eval(model, train_loader, test_loader, device))
    knn5 = float(knn_eval(model, train_loader, test_loader, device, k=5))
    emb, flat = get_enc_inputs(model, test_loader, device)
    ordering = compute_ordering_scores(emb, flat)

    result = {
        "point": point, "dataset": dataset, "seed": seed,
        "params": POINTS[point],
        "ordering_variance_decay": ordering["variance_decay_rho"],
        "linear_probe": lp, "knn5": knn5,
        "epoch_reached_128": epoch_128,
        "reached_128": epoch_128 is not None,
        "n_growth_events": len(events),
        "growth_schedule": events,
        "final_dim": int(getattr(model, "latent_dim", MAX_DIM)),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    fname = OUT / f"{dataset}_{point}_seed{seed}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    reached = epoch_128 if epoch_128 is not None else "NOT-REACHED"
    print(f"  ordering={ordering['variance_decay_rho']:.3f}  LP={lp:.4f}  "
          f"reached128@{reached}  -> {fname.name}")
    return result


def summarize():
    rows = {}
    for fp in glob.glob(str(OUT / "*.json")):
        with open(fp) as f:
            r = json.load(f)
        rows.setdefault((r["dataset"], r["point"]), []).append(r)

    order = {"fast": 0, "mid": 1, "slow": 2}
    print("\n" + "=" * 84)
    print("ID-TRIGGER FRONTIER (mean +/- std across seeds)")
    print("=" * 84)
    print(f"{'Dataset':<10}{'Point':<7}{'n':>3}{'Ordering':>16}{'LinProbe':>16}"
          f"{'Epoch->128':>12}{'reached/n':>12}")
    print("-" * 84)
    for (ds, pt), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
        n = len(rs)
        ordv = np.array([r["ordering_variance_decay"] for r in rs])
        lp = np.array([r["linear_probe"] for r in rs])
        reached = [r for r in rs if r["reached_128"]]
        e128 = [r["epoch_reached_128"] for r in reached]
        e128_str = f"{np.mean(e128):.1f}" if e128 else "-"
        print(f"{ds:<10}{pt:<7}{n:>3}"
              f"{ordv.mean():>9.3f}+/-{ordv.std():.3f}"
              f"{lp.mean():>9.4f}+/-{lp.std():.4f}"
              f"{e128_str:>12}{f'{len(reached)}/{n}':>12}")
    print("-" * 84)
    print("\nFor the frontier figure: plot each point's (mean epoch->128, mean")
    print("ordering) with std bars; overlay FIXED_GRID points from Week-2.")
    print("Any point with reached/n < n has seeds that stalled - check before")
    print("plotting (a valid frontier point reaches 128 on all seeds).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--point", choices=list(POINTS), default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summarize()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.point is not None and args.seed is not None:
        run_cell(args.point, args.dataset, args.seed, device)
        return

    loaders = build_dataloaders(base_cfg(args.dataset, 0))
    points = [args.point] if args.point else list(POINTS)
    for pt in points:
        for seed in SEEDS:
            fname = OUT / f"{args.dataset}_{pt}_seed{seed}.json"
            if fname.exists():
                print(f"skip (exists): {fname.name}")
                continue
            run_cell(pt, args.dataset, seed, device, loaders=loaders)

    summarize()


if __name__ == "__main__":
    main()
