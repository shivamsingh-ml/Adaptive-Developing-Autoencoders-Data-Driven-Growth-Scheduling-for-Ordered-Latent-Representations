"""
run_ablation_grid.py  —  Week-2 full growth-schedule ablation

Sweeps the programmatic schedule grid (growth_rate x epochs_per_stage)
across datasets and seeds, saving each run to results/raw_ablation/.

Includes a CONFIRMATION GATE that runs first: the fast-growth config
at 3 seeds. If the ordering sign is unstable across those seeds, the
script pauses and reports instead of burning the full grid.

Usage:
    # Step 1 — run the gate only:
    python run_ablation_grid.py --gate_only

    # Step 2 — if gate passes, run the full grid:
    python run_ablation_grid.py --dataset cifar10
    python run_ablation_grid.py --dataset cifar100

    # Or a single cell (for SLURM array jobs):
    python run_ablation_grid.py --dataset cifar10 --rate 1.7 --eps 8 --seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import build_dataloaders
from models.conv_autoencoder import ConvAutoencoder
from solver import train_with_trigger
from triggers.programmatic_trigger import ProgrammaticTrigger
from evaluation.linear_probe import linear_probe_eval
from evaluation.knn_eval import knn_eval
from evaluation.intrinsic_dim import twonn_intrinsic_dim
from evaluation.ordering_score import compute_ordering_scores


RATES = [1.3, 1.5, 1.7, 2.0, 2.5]
EPOCHS_PER_STAGE = [4, 6, 8, 10]
SEEDS = [0, 1, 2, 3, 4]
TOTAL_EPOCHS = 60
START_DIM, MAX_DIM = 6, 128

OUT = Path("results/raw_ablation")


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def base_cfg(dataset, seed):
    return {
        "experiment": {"name": "ablation", "seed": seed, "device": "cuda"},
        "data": {"dataset": dataset, "data_dir": "./data",
                 "batch_size": 128, "num_workers": 0},
        "model": {"type": "conv_autoencoder", "start_dim": START_DIM, "max_dim": MAX_DIM},
        "training": {"epochs": TOTAL_EPOCHS, "lr": 0.1, "optimizer": "sgd", "loss": "mse"},
    }


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


def run_cell(dataset, rate, eps, seed, device, loaders=None):
    cfg = base_cfg(dataset, seed)
    if loaders is None:
        loaders = build_dataloaders(cfg)
    train_loader, val_loader, test_loader = loaders

    set_seed(seed)
    trig = ProgrammaticTrigger(START_DIM, MAX_DIM, rate, eps, TOTAL_EPOCHS)
    model = ConvAutoencoder(latent_dim=trig.current_dim).to(device)

    print(f"\n{'='*70}")
    print(f"ABLATION | {dataset} | rate={rate} eps/stage={eps} seed={seed}")
    print(f"{'='*70}")

    model, history = train_with_trigger(
        model=model, trigger=trig, model_cls=ConvAutoencoder,
        train_loader=train_loader, val_loader=val_loader,
        optimizer_cls=optim.SGD, loss_fn=nn.MSELoss(),
        cfg=cfg, device=device,
    )

    lp = float(linear_probe_eval(model, train_loader, test_loader, device))
    knn5 = float(knn_eval(model, train_loader, test_loader, device, k=5))
    emb, flat = get_enc_inputs(model, test_loader, device)
    ordering = compute_ordering_scores(emb, flat)
    fid = float(twonn_intrinsic_dim(emb))

    result = {
        "dataset": dataset, "rate": rate, "epochs_per_stage": eps, "seed": seed,
        "linear_probe": lp, "knn5": knn5,
        "ordering_variance_decay": ordering["variance_decay_rho"],
        "ordering_pc_alignment": ordering["pc_alignment_rho"],
        "final_intrinsic_dim": fid,
        "final_dim": int(getattr(model, "latent_dim", MAX_DIM)),
        "reached_naturally": trig.reached_naturally,
        "n_growth_events": sum(1 for i in range(1, len(history))
                               if history[i]["latent_dim"] != history[i-1]["latent_dim"]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    fname = f"{dataset}_rate{rate}_eps{eps}_seed{seed}.json"
    with open(OUT / fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  LP={lp:.4f}  ordering={ordering['variance_decay_rho']:.3f}  -> {fname}")
    return result


def run_gate(device):
    """Run fast config (rate=2.5, eps=4) at 3 seeds; check ordering sign stability."""
    print("\n" + "#" * 70)
    print("# CONFIRMATION GATE: fast config (rate=2.5, eps=4) at seeds 0,1,2")
    print("#" * 70)
    signs = []
    for seed in [0, 1, 2]:
        r = run_cell("cifar10", 2.5, 4, seed, device)
        signs.append(r["ordering_variance_decay"])

    print("\n" + "=" * 70)
    print("GATE RESULT")
    print("=" * 70)
    for s, v in zip([0, 1, 2], signs):
        print(f"  seed {s}: ordering = {v:+.3f}")
    all_neg = all(v < 0 for v in signs)
    all_pos = all(v > 0 for v in signs)
    if all_neg or all_pos:
        print(f"\n  PASS — sign is stable ({'negative' if all_neg else 'positive'}).")
        print("  Safe to run the full grid.")
        return True
    else:
        print("\n  FAIL — ordering sign is UNSTABLE across seeds.")
        print("  Do NOT run the full grid yet. The fast regime is noisy;")
        print("  consider more seeds or revisiting the metric before committing.")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_only", action="store_true")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--eps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.gate_only:
        run_gate(device)
        return

    # Single-cell mode (for SLURM arrays)
    if args.rate is not None and args.eps is not None and args.seed is not None:
        run_cell(args.dataset or "cifar10", args.rate, args.eps, args.seed, device)
        return

    # Full sweep for one dataset (sequential)
    ds = args.dataset or "cifar10"
    loaders = build_dataloaders(base_cfg(ds, 0))  # split is seed-fixed (42) in data.py
    for rate in RATES:
        for eps in EPOCHS_PER_STAGE:
            for seed in SEEDS:
                # skip if already done (resumability)
                fname = OUT / f"{ds}_rate{rate}_eps{eps}_seed{seed}.json"
                if fname.exists():
                    print(f"skip (exists): {fname.name}")
                    continue
                run_cell(ds, rate, eps, seed, device, loaders=loaders)


if __name__ == "__main__":
    main()
