"""
run_trigger_smoketest.py  —  Week-3 GATE before full adaptive-trigger runs

Runs ID-trigger and GV-trigger for 1 seed each on CIFAR-10 and reports the
produced growth schedule, epoch-reached-128, ordering/LP/kNN, and the
per-epoch trigger signal.

NOTE: the ordering>0.85 line is a convenience sanity check, NOT the verdict.
The real evaluation is the 5-seed mean and the ordering-vs-epoch frontier.

Run from project root:
    python run_trigger_smoketest.py
"""

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
from evaluation.ordering_score import compute_ordering_scores


TOTAL_EPOCHS = 60
START_DIM, MAX_DIM = 6, 128
FIXED_REF_EPOCH = 48


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def base_cfg(seed=0):
    return {
        "experiment": {"name": "smoketest", "seed": seed, "device": "cuda"},
        "data": {"dataset": "cifar10", "data_dir": "./data",
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


def growth_schedule_from_history(history):
    events = []
    for i in range(1, len(history)):
        if history[i]["latent_dim"] != history[i-1]["latent_dim"]:
            events.append((history[i]["epoch"],
                           history[i-1]["latent_dim"],
                           history[i]["latent_dim"]))
    return events


def run_trigger(name, trigger, signal_key, cfg, device, loaders):
    train_loader, val_loader, test_loader = loaders
    set_seed(cfg["experiment"]["seed"])
    model = ConvAutoencoder(latent_dim=trigger.current_dim).to(device)

    print(f"\n{'='*70}\nSMOKE TEST: {name}\n{'='*70}")
    model, history = train_with_trigger(
        model=model, trigger=trigger, model_cls=ConvAutoencoder,
        train_loader=train_loader, val_loader=val_loader,
        optimizer_cls=optim.SGD, loss_fn=nn.MSELoss(),
        cfg=cfg, device=device,
    )

    events = growth_schedule_from_history(history)
    epoch_128 = next((ep for ep, o, n in events if n == 128), None)
    lp = float(linear_probe_eval(model, train_loader, test_loader, device))
    knn5 = float(knn_eval(model, train_loader, test_loader, device, k=5))
    emb, flat = get_enc_inputs(model, test_loader, device)
    ordering = compute_ordering_scores(emb, flat)["variance_decay_rho"]

    signal = [h.get(signal_key) for h in history]

    print(f"\n--- {name} RESULTS ---")
    print("Growth schedule (epoch: old->new):")
    for ep, o, n in events:
        print(f"    epoch {ep:2d}: {o:3d} -> {n:3d}")
    print(f"Reached 128 at epoch: {epoch_128}")
    print(f"N growth events:      {len(events)}")
    print(f"Ordering rho:         {ordering:.3f}")
    print(f"Linear probe:         {lp:.4f}")
    print(f"kNN-5:                {knn5:.4f}")

    print(f"\n{signal_key} curve (every 3 epochs):")
    for e in range(0, len(signal), 3):
        v = signal[e]
        vs = f"{v:.3e}" if isinstance(v, (int, float)) and v is not None else "-"
        dim = history[e]["latent_dim"]
        print(f"    ep {e:2d}: {signal_key}={vs:>8}  dim={dim}")

    print(f"\n--- {name} SANITY CHECKS ---")
    c1 = (epoch_128 is not None)
    c2 = (len(events) >= 4)
    c3 = (ordering > 0.85)
    c4 = (epoch_128 is not None and epoch_128 < FIXED_REF_EPOCH)
    print(f"  [{'OK ' if c1 else 'NO '}] reaches dim=128")
    print(f"  [{'OK ' if c2 else 'NO '}] fires multiple times: {len(events)} events")
    print(f"  [{'OK ' if c3 else '~~ '}] ordering > 0.85 (convenience cutoff): {ordering:.3f}")
    print(f"  [{'OK ' if c4 else 'NO '}] reaches 128 before epoch {FIXED_REF_EPOCH}: {epoch_128}")
    print(f"  (verdict is the 5-seed mean + frontier, not this single seed)")

    return {
        "name": name, "events": events, "epoch_128": epoch_128,
        "ordering": ordering, "linear_probe": lp, "knn5": knn5,
        "signal": signal,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = base_cfg(seed=0)
    loaders = build_dataloaders(cfg)

    results = []

    id_trig = IDTrigger(START_DIM, MAX_DIM, growth_rate=1.7,
                        delta_threshold=0.3, patience=3,
                        min_epochs_per_stage=3, smooth_window=3,
                        max_epochs_per_stage=11)
    results.append(run_trigger("ID-trigger", id_trig, "intrinsic_dim",
                               cfg, device, loaders))

    gv_trig = GVTrigger(START_DIM, MAX_DIM, growth_rate=1.7,
                        relative_threshold=0.4, patience=2,
                        peak_floor=1e-9, min_epochs_per_stage=3,
                        max_epochs_per_stage=11)
    results.append(run_trigger("GV-trigger", gv_trig, "gradient_variance",
                               cfg, device, loaders))

    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['name']:<12} ordering={r['ordering']:.3f}  "
              f"reached128@{r['epoch_128']}")

    Path("results/processed").mkdir(parents=True, exist_ok=True)
    # keep the per-epoch signal as 'id_curve' so plot_trigger_signal.py can read it;
    # also expose it under 'signal' and keep 'growth_schedule' for the same tool.
    saved = []
    for r in results:
        rec = {k: v for k, v in r.items() if k != "signal"}
        rec["id_curve"] = r["signal"]
        rec["signal"] = r["signal"]
        rec["growth_schedule"] = r["events"]
        saved.append(rec)
    with open("results/processed/trigger_smoketest.json", "w") as f:
        json.dump(saved, f, indent=2)
    print("\nSaved results/processed/trigger_smoketest.json")


if __name__ == "__main__":
    main()