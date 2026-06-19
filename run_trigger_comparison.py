"""
run_trigger_comparison.py  —  Week-3/4 full adaptive-trigger evaluation

Runs the adaptive triggers (ID and GV) across multiple seeds and datasets,
logging everything needed to place them on the ordering-vs-efficiency
frontier against the fixed-schedule grid:
  - ordering (variance-decay), linear probe, kNN
  - epoch at which dim=128 was reached
  - the growth schedule the trigger produced
  - n_growth_events

Resumable: skips any (trigger, dataset, seed) whose JSON already exists.

Usage:
    python run_trigger_comparison.py --dataset cifar10
    python run_trigger_comparison.py --dataset cifar10 --trigger id          # ID only
    python run_trigger_comparison.py --dataset cifar10 --trigger id --seed 0 # single cell
    python run_trigger_comparison.py --summary
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
from triggers.gv_trigger import GVTrigger
from evaluation.linear_probe import linear_probe_eval
from evaluation.knn_eval import knn_eval
from evaluation.intrinsic_dim import twonn_intrinsic_dim
from evaluation.ordering_score import compute_ordering_scores


TOTAL_EPOCHS = 60
START_DIM, MAX_DIM = 6, 128
SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["cifar10", "cifar100"]
OUT = Path("results/raw_triggers")

# Fixed-schedule frontier anchors from the Week-2 grid (cheapest safe config).
# Used to place the adaptive triggers on the ordering-vs-epoch frontier.
FIXED_REFERENCE = {
    "cifar10":  {"config": "rate1.7/eps8", "ordering": 0.88, "epoch_128": 48},
    "cifar100": {"config": "rate1.7/eps8", "ordering": 0.90, "epoch_128": 48},
}


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def base_cfg(dataset, seed):
    return {
        "experiment": {"name": "trigger", "seed": seed, "device": "cuda"},
        "data": {"dataset": dataset, "data_dir": "./data",
                 "batch_size": 128, "num_workers": 0},
        "model": {"type": "conv_autoencoder", "start_dim": START_DIM, "max_dim": MAX_DIM},
        "training": {"epochs": TOTAL_EPOCHS, "lr": 0.1, "optimizer": "sgd", "loss": "mse"},
    }


def make_trigger(kind):
    if kind == "id":
        # v4 committed "fast" config
        return IDTrigger(START_DIM, MAX_DIM, growth_rate=1.7,
                         delta_threshold=0.3, patience=3,
                         min_epochs_per_stage=3, smooth_window=3,
                         max_epochs_per_stage=11)
    elif kind == "gv":
        return GVTrigger(START_DIM, MAX_DIM, growth_rate=1.7,
                         relative_threshold=0.05, patience=3,
                         min_epochs_per_stage=3, max_epochs_per_stage=11)
    raise ValueError(kind)


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


def run_cell(kind, dataset, seed, device, loaders=None):
    cfg = base_cfg(dataset, seed)
    if loaders is None:
        loaders = build_dataloaders(cfg)
    train_loader, val_loader, test_loader = loaders

    set_seed(seed)
    trig = make_trigger(kind)
    model = ConvAutoencoder(latent_dim=trig.current_dim).to(device)

    print(f"\n{'='*70}\n{kind.upper()}-trigger | {dataset} | seed {seed}\n{'='*70}")
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
    fid = float(twonn_intrinsic_dim(emb))

    result = {
        "trigger": kind, "dataset": dataset, "seed": seed,
        "ordering_variance_decay": ordering["variance_decay_rho"],
        "ordering_pc_alignment": ordering["pc_alignment_rho"],
        "linear_probe": lp, "knn5": knn5, "final_intrinsic_dim": fid,
        "epoch_reached_128": epoch_128,
        "n_growth_events": len(events),
        "growth_schedule": events,
        "final_dim": int(getattr(model, "latent_dim", MAX_DIM)),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    fname = OUT / f"{dataset}_{kind}_seed{seed}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  ordering={ordering['variance_decay_rho']:.3f}  LP={lp:.4f}  "
          f"reached128@{epoch_128}  -> {fname.name}")
    return result


def summarize():
    rows = {}
    for fp in glob.glob(str(OUT / "*.json")):
        with open(fp) as f:
            r = json.load(f)
        key = (r["dataset"], r["trigger"])
        rows.setdefault(key, []).append(r)

    print("\n" + "=" * 78)
    print("ADAPTIVE TRIGGER SUMMARY (mean +/- std across seeds)")
    print("=" * 78)
    print(f"{'Dataset':<10}{'Trigger':<9}{'n':>3}{'Ordering':>16}{'LinProbe':>16}"
          f"{'Epoch->128':>12}")
    print("-" * 78)
    for (ds, kind), rs in sorted(rows.items()):
        n = len(rs)
        ordv = np.array([r["ordering_variance_decay"] for r in rs])
        lp = np.array([r["linear_probe"] for r in rs])
        e128 = [r["epoch_reached_128"] for r in rs if r["epoch_reached_128"]]
        e128_str = f"{np.mean(e128):.1f}" if e128 else "-"
        print(f"{ds:<10}{kind:<9}{n:>3}"
              f"{ordv.mean():>9.3f}+/-{ordv.std():.3f}"
              f"{lp.mean():>9.4f}+/-{lp.std():.4f}"
              f"{e128_str:>12}")

    print("-" * 78)
    print("\nFIXED-SCHEDULE FRONTIER ANCHOR (from Week-2 grid):")
    for ds, ref in FIXED_REFERENCE.items():
        print(f"  {ds}: {ref['config']} ordering~{ref['ordering']}, "
              f"reaches 128 ~epoch {ref['epoch_128']}")
    print("\nFRAMING: the trigger is a point on the ordering-vs-efficiency")
    print("frontier. Compare (ordering, epoch->128) against the fixed grid;")
    print("the claim is matching the frontier WITHOUT a schedule search.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--trigger", choices=["id", "gv"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summarize()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.trigger is not None and args.seed is not None:
        run_cell(args.trigger, args.dataset, args.seed, device)
        return

    loaders = build_dataloaders(base_cfg(args.dataset, 0))
    triggers = [args.trigger] if args.trigger else ["id", "gv"]
    for kind in triggers:
        for seed in SEEDS:
            fname = OUT / f"{args.dataset}_{kind}_seed{seed}.json"
            if fname.exists():
                print(f"skip (exists): {fname.name}")
                continue
            run_cell(kind, args.dataset, seed, device, loaders=loaders)

    summarize()


if __name__ == "__main__":
    main()