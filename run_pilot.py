"""
run_pilot.py  —  Week-2 schedule-ablation PILOT (de-risking step)

Runs 6 configs (1 seed each) that span both schedule-generation
approaches at their extremes, then prints a comparison table so you
can decide whether growth schedule meaningfully affects ordering.

Approach A (programmatic): schedule derived from (growth_rate, epochs_per_stage)
Approach B (fixed):        hand-written full schedule lists

DECISION RULE (printed at the end):
  - If ordering (variance-decay) range across configs > 0.15
        -> schedule matters -> run the full grid
  - If everything clusters tight (range < 0.15)
        -> schedule is robust -> pivot paper emphasis to adaptive triggers

Run from project root:
    python run_pilot.py --dataset cifar10 --seed 0
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
from triggers.fixed_trigger import FixedTrigger
from triggers.programmatic_trigger import ProgrammaticTrigger

from evaluation.linear_probe import linear_probe_eval
from evaluation.ordering_score import compute_ordering_scores


# ── Pilot config definitions ──────────────────────────────────────────────────
PROGRAMMATIC_CONFIGS = [
    # (name, growth_rate, epochs_per_stage)
    ("A_slow",       1.3, 10),
    ("A_paper_like", 1.7, 8),
    ("A_fast",       2.5, 4),
]

FIXED_CONFIGS = {
    "B_slow":  [6]*10 + [10]*10 + [17]*10 + [29]*10 + [50]*8 + [85]*7 + [128]*5,
    "B_paper": [6]*6 + [10]*6 + [17]*7 + [29]*7 + [50]*8 + [85]*8 + [128]*18,
    "B_fast":  [6]*2 + [17]*2 + [50]*3 + [85]*3 + [128]*50,
}

TOTAL_EPOCHS = 60
START_DIM = 6
MAX_DIM = 128


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def base_cfg(dataset, seed):
    return {
        "experiment": {"name": "pilot", "seed": seed, "device": "cuda"},
        "data": {"dataset": dataset, "data_dir": "./data",
                 "batch_size": 128, "num_workers": 0},
        "model": {"type": "conv_autoencoder", "start_dim": START_DIM, "max_dim": MAX_DIM},
        "training": {"epochs": TOTAL_EPOCHS, "lr": 0.1, "optimizer": "sgd", "loss": "mse"},
    }


@torch.no_grad()
def get_encodings_and_inputs(model, loader, device, max_samples=2000):
    model.eval()
    embs, flats = [], []
    n = 0
    for x, _ in loader:
        z = model.encode(x.to(device)).detach().cpu()
        embs.append(z)
        flats.append(x.view(x.size(0), -1))
        n += x.size(0)
        if n >= max_samples:
            break
    embs = torch.cat(embs)[:max_samples].numpy()
    flats = torch.cat(flats)[:max_samples].numpy()
    return embs, flats


def run_one(name, trigger, cfg, device, loaders):
    train_loader, val_loader, test_loader = loaders
    set_seed(cfg["experiment"]["seed"])

    latent_dim = trigger.current_dim
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)

    print(f"\n{'='*70}\nPILOT RUN: {name}  (start dim={latent_dim})\n{'='*70}")
    model, history = train_with_trigger(
        model=model, trigger=trigger, model_cls=ConvAutoencoder,
        train_loader=train_loader, val_loader=val_loader,
        optimizer_cls=optim.SGD, loss_fn=nn.MSELoss(),
        cfg=cfg, device=device,
    )

    lp = float(linear_probe_eval(model, train_loader, test_loader, device))
    emb, flat = get_encodings_and_inputs(model, test_loader, device)
    ordering = compute_ordering_scores(emb, flat)

    final_dim = int(getattr(model, "latent_dim", latent_dim))
    n_growth = sum(1 for i in range(1, len(history))
                   if history[i]["latent_dim"] != history[i-1]["latent_dim"])
    reached_naturally = getattr(trigger, "reached_naturally", None)

    return {
        "name": name,
        "linear_probe": lp,
        "ordering_variance_decay": ordering["variance_decay_rho"],
        "ordering_pc_alignment": ordering["pc_alignment_rho"],
        "final_dim": final_dim,
        "n_growth_events": n_growth,
        "reached_naturally": reached_naturally,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = base_cfg(args.dataset, args.seed)
    loaders = build_dataloaders(cfg)

    results = []

    # Approach A — programmatic
    for name, rate, eps in PROGRAMMATIC_CONFIGS:
        trig = ProgrammaticTrigger(START_DIM, MAX_DIM, rate, eps, TOTAL_EPOCHS)
        results.append(run_one(name, trig, cfg, device, loaders))

    # Approach B — fixed hand-made
    for name, schedule in FIXED_CONFIGS.items():
        trig = FixedTrigger(schedule=schedule)
        results.append(run_one(name, trig, cfg, device, loaders))

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 78)
    print("PILOT RESULTS")
    print("=" * 78)
    print(f"{'Config':<14}{'LinProbe':>10}{'Ordering':>11}{'FinalDim':>10}"
          f"{'Growths':>9}{'Natural':>9}")
    print("-" * 78)
    for r in results:
        nat = "" if r["reached_naturally"] is None else str(r["reached_naturally"])
        print(f"{r['name']:<14}{r['linear_probe']:>10.4f}"
              f"{r['ordering_variance_decay']:>11.3f}{r['final_dim']:>10}"
              f"{r['n_growth_events']:>9}{nat:>9}")

    ord_vals = [r["ordering_variance_decay"] for r in results]
    lp_vals = [r["linear_probe"] for r in results]
    ord_range = max(ord_vals) - min(ord_vals)
    lp_range = max(lp_vals) - min(lp_vals)

    print("-" * 78)
    print(f"Ordering range across configs: {ord_range:.3f}")
    print(f"Linear-probe range across configs: {lp_range:.4f}")
    print()
    print("DECISION:")
    if ord_range > 0.15 or lp_range > 0.03:
        print(f"  Schedule MATTERS (ordering range {ord_range:.3f}, "
              f"LP range {lp_range:.4f}).")
        print("  -> Proceed with the full Week-2 ablation grid.")
    else:
        print(f"  Schedule appears ROBUST (ordering range {ord_range:.3f}, "
              f"LP range {lp_range:.4f}).")
        print("  -> Consider pivoting paper emphasis to adaptive TRIGGERS.")
        print("     A short robustness section replaces the full grid.")

    out = Path("results/processed"); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"pilot_{args.dataset}_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results/processed/pilot_{args.dataset}_seed{args.seed}.json")


if __name__ == "__main__":
    main()
