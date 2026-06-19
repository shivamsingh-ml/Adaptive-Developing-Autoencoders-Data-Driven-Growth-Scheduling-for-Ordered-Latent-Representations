import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from data import build_dataloaders
from evaluation.intrinsic_dim import twonn_intrinsic_dim, twonn_intrinsic_dim_exact
from evaluation.knn_eval import knn_eval
from evaluation.linear_probe import linear_probe_eval
from models.conv_autoencoder import ConvAutoencoder
from solver import train_with_trigger
from evaluation.ordering_score import compute_ordering_scores
from triggers import build_trigger


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=5000):
    model.eval()

    xs, ys = [], []
    n_seen = 0

    pbar = tqdm(loader, desc="Extract embeddings", leave=False)

    for x, y in pbar:
        x = x.to(device)
        z = model.encode(x).detach().cpu()

        xs.append(z)
        ys.append(y.cpu())

        n_seen += z.shape[0]
        if n_seen >= max_samples:
            break

    x_all = torch.cat(xs, dim=0)[:max_samples]
    y_all = torch.cat(ys, dim=0)[:max_samples]

    return x_all.numpy(), y_all.numpy()


def count_growth_events(history):
    return sum(
        1
        for i in range(1, len(history))
        if history[i]["latent_dim"] != history[i - 1]["latent_dim"]
    )


def get_growth_epochs(history):
    return [
        history[i]["epoch"]
        for i in range(1, len(history))
        if history[i]["latent_dim"] != history[i - 1]["latent_dim"]
    ]


def save_run_json(cfg, history, metrics):
    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{cfg['data']['dataset']}_"
        f"{cfg['experiment']['name']}_"
        f"seed{cfg['experiment'].get('seed', 0)}"
    )

    payload = {
        "run_id": run_id,
        "seed": cfg["experiment"].get("seed", 0),
        "dataset": cfg["data"]["dataset"],
        "experiment": cfg["experiment"]["name"],
        "model": cfg["model"],
        "trigger": cfg["trigger"],
        "training": cfg["training"],
        "evaluation": cfg.get("evaluation", {}),
        "history": history,
        **metrics,
    }

    path = out_dir / f"{run_id}.json"

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results to {path}")


def print_experiment_header(cfg, device, latent_dim):
    print("=" * 80)
    print(f"Experiment:       {cfg['experiment']['name']}")
    print(f"Dataset:          {cfg['data']['dataset']}")
    print(f"Seed:             {cfg['experiment'].get('seed', 0)}")
    print(f"Device:           {device}")
    print(f"Epochs:           {cfg['training']['epochs']}")
    print(f"Learning rate:    {cfg['training']['lr']}")
    print(f"Initial latent:   {latent_dim}")
    print(f"Trigger:          {cfg['trigger']['type']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    if args.dataset is not None:
        cfg["data"]["dataset"] = args.dataset

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    if args.lr is not None:
        cfg["training"]["lr"] = args.lr

    set_seed(cfg["experiment"].get("seed", 0))

    device = torch.device(
        cfg["experiment"].get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    trigger = build_trigger(cfg["trigger"])

    latent_dim = trigger.current_dim
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)

    print_experiment_header(cfg, device, latent_dim)

    model, history = train_with_trigger(
        model=model,
        trigger=trigger,
        model_cls=ConvAutoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_cls=optim.SGD,
        loss_fn=nn.MSELoss(),
        cfg=cfg,
        device=device,
    )

    print("\nRunning evaluation...")

    metrics = {}
    eval_cfg = cfg.get("evaluation", {})

    if eval_cfg.get("linear_probe", True):
        print("Evaluating linear probe...")
        metrics["linear_probe_acc"] = float(
            linear_probe_eval(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                solver=eval_cfg.get("linear_probe_solver", "lbfgs"),
            )
        )
        print(f"Linear probe accuracy: {metrics['linear_probe_acc']:.4f}")

    if eval_cfg.get("knn", True):
        print("Evaluating k-NN...")
        metrics["knn5_acc"] = float(
            knn_eval(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                k=5,
            )
        )
        print(f"k-NN@5 accuracy: {metrics['knn5_acc']:.4f}")

    if eval_cfg.get("intrinsic_dim", True):
        print("Computing final intrinsic dimension...")
        x_emb, _ = extract_embeddings(
            model=model,
            loader=test_loader,
            device=device,
            max_samples=5000,
        )
        metrics["final_intrinsic_dim"] = float(twonn_intrinsic_dim(x_emb))
        metrics["final_intrinsic_dim_Facco"] = float(twonn_intrinsic_dim_exact(x_emb))
        print(f"Final intrinsic dimension: {metrics['final_intrinsic_dim']:.4f}")


    if eval_cfg.get("ordering_score", False):
        print("Computing ordering scores...")
        # Get bottleneck encodings + flattened inputs on the same samples
        x_emb, _ = extract_embeddings(model, test_loader, device, max_samples=2000)
        # Collect the matching flattened input images
        flat_inputs = []
        n_seen = 0
        for x, _ in test_loader:
            flat_inputs.append(x.view(x.size(0), -1).numpy())
            n_seen += x.size(0)
            if n_seen >= 2000:
                break
        flat_inputs = np.concatenate(flat_inputs, axis=0)[:2000]
        ord_scores = compute_ordering_scores(x_emb, flat_inputs)
        metrics["ordering_pc_alignment"] = ord_scores["pc_alignment_rho"]
        metrics["ordering_variance_decay"] = ord_scores["variance_decay_rho"]
        print(f"  PC-alignment rho: {metrics['ordering_pc_alignment']:.3f}")
        print(f"  Variance-decay rho: {metrics['ordering_variance_decay']:.3f}")

    metrics["final_latent_dim"] = int(getattr(model, "latent_dim", latent_dim))
    metrics["n_growth_events"] = count_growth_events(history)
    metrics["growth_epochs"] = get_growth_epochs(history)

    print("\nFinal summary:")
    print(f"Final latent dim:  {metrics['final_latent_dim']}")
    print(f"Growth events:     {metrics['n_growth_events']}")
    print(f"Growth epochs:     {metrics['growth_epochs']}")

    save_run_json(cfg, history, metrics)


if __name__ == "__main__":
    main()