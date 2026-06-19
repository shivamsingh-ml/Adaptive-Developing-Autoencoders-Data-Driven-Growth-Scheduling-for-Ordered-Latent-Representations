import numpy as np
import torch
from tqdm import tqdm
from evaluation.intrinsic_dim import twonn_intrinsic_dim

@torch.no_grad()
def compute_epoch_id(model, val_loader, device, max_samples=2000):
    """Fast subsampled TwoNN ID on val embeddings, for per-epoch logging."""
    model.eval()
    embs = []
    n = 0
    for x, _ in val_loader:
        x = x.to(device)
        z = model.encode(x).detach().cpu()
        embs.append(z)
        n += z.shape[0]
        if n >= max_samples:
            break
    embs = torch.cat(embs, dim=0)[:max_samples].numpy()
    # Guard: ID needs at least latent_dim distinct points
    if embs.shape[0] < 10:
        return float("nan")
    return float(twonn_intrinsic_dim(embs))



def compute_bottleneck_gradient_variance(model):
    grads = []

    for name, param in model.named_parameters():
        if name in ("encoder.11.weight", "encoder.11.bias", "linear.0.weight") and param.grad is not None:
            grads.append(param.grad.detach().flatten())

    if not grads:
        return 0.0

    return torch.cat(grads).var(unbiased=False).item()


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch=None):
    model.train()

    losses = []
    batch_grad_vars = []

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}" if epoch is not None else "Train",
        leave=False,
    )

    for x, _ in pbar:
        x = x.to(device)

        optimizer.zero_grad()
        _, x_hat = model(x)
        loss = loss_fn(x_hat, x)
        loss.backward()

        grad_var = compute_bottleneck_gradient_variance(model)
        batch_grad_vars.append(grad_var)

        optimizer.step()

        losses.append(loss.item())

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            grad_var=f"{grad_var:.2e}",
        )

    return {
        "train_loss": float(np.mean(losses)) if losses else 0.0,
        "gradient_variance": float(np.mean(batch_grad_vars)) if batch_grad_vars else 0.0,
    }


def validate_one_epoch(model, val_loader, loss_fn, device):
    model.eval()

    losses = []

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)

            _, x_hat = model(x)
            loss = loss_fn(x_hat, x)

            losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def train_with_trigger(
    model,
    trigger,
    model_cls,
    train_loader,
    val_loader,
    optimizer_cls,
    loss_fn,
    cfg,
    device,
):
    from models.growth import grow_model

    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["lr"]
    freeze_until = cfg["training"].get("freeze_until_epoch", 0)  # NEW

    # NEW: freeze bottleneck if PCA-AE
    if freeze_until > 0:
        model.encoder[11].weight.requires_grad = False
        model.encoder[11].bias.requires_grad = False

    optimizer = optimizer_cls(model.parameters(), lr=lr)

    history = []

    epoch_bar = tqdm(range(epochs), desc="Training")

    for epoch in epoch_bar:
        # NEW: unfreeze bottleneck at freeze_until epoch
        if freeze_until > 0 and epoch == freeze_until:
            model.encoder[11].weight.requires_grad = True
            model.encoder[11].bias.requires_grad = True
            optimizer = optimizer_cls(
                [p for p in model.parameters() if p.requires_grad], lr=lr
            )
            print(f"\n[PCA-AE] Unfroze bottleneck at epoch {epoch}")
            
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
        )

        val_loss = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        # NEW: per-epoch intrinsic dimensionality (subsampled, fast)
        epoch_id = compute_epoch_id(model, val_loader, device, max_samples=2000)

        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_loss,
            "gradient_variance": train_metrics["gradient_variance"],
            "latent_dim": trigger.current_dim,
            "intrinsic_dim": epoch_id,
        }

        history.append(metrics)

        epoch_bar.set_postfix(
            train=f"{metrics['train_loss']:.4f}",
            val=f"{metrics['val_loss']:.4f}",
            grad_var=f"{metrics['gradient_variance']:.2e}",
            dim=metrics["latent_dim"],
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train {metrics['train_loss']:.4f} | "
            f"Val {metrics['val_loss']:.4f} | "
            f"GradVar {metrics['gradient_variance']:.3e} | "
            f"Latent {metrics['latent_dim']}"
        )
        trigger.step_epoch()
        if trigger.should_grow(metrics):
            old_dim = trigger.current_dim
            new_dim = trigger.next_dim(metrics)

            print(
                "\n"
                f"GROWTH EVENT | "
                f"Epoch={epoch + 1} | "
                f"{old_dim} -> {new_dim}"
            )
            
            model = grow_model(
                old_model=model,
                model_cls=model_cls,
                new_latent_dim=new_dim,
                device=device,
            )

            optimizer = optimizer_cls(model.parameters(), lr=lr)
    return model, history