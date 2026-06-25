import os
import random
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt

from .forward import forward_diffusion_pair
from .loss import diffusion_noise_loss


def train_one_epoch(model, dataloader, schedule, optimizer, device, loss_type="l2", grad_clip=1.0):
    """One pass over the dataset; returns mean loss."""
    model.train()
    epoch_losses = []
    for x0 in dataloader:
        x0 = x0.to(device)
        x_t, t, noise = forward_diffusion_pair(x0, schedule)
        noise_pred = model(x_t, t)
        loss = diffusion_noise_loss(noise, noise_pred, loss_type)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        epoch_losses.append(loss.item())
    return float(np.mean(epoch_losses))


def train(
    model,
    dataloader,
    schedule,
    optimizer,
    device,
    epochs: int,
    save_dir: str,
    results_dir: str,
    loss_type: str = "l2",
    save_every: int = 100,
    log_every: int = 10,
    config: dict = None,
    grad_clip: float = 1.0,
    scheduler=None,
) -> List[float]:
    """Full training loop with periodic checkpoints and a loss plot."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    config = config or {}

    loss_history = []
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, schedule, optimizer, device, loss_type, grad_clip)
        if scheduler is not None:
            scheduler.step()
        loss_history.append(avg_loss)
        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | loss = {avg_loss:.6f}")
        if epoch % save_every == 0:
            save_checkpoint(
                Path(save_dir) / "diffusion_model.pt",
                model, optimizer, schedule, epoch, loss_history, config,
            )

    save_checkpoint(
        Path(save_dir) / "diffusion_model.pt",
        model, optimizer, schedule, epochs, loss_history, config,
    )
    plot_loss_curve(loss_history, Path(results_dir) / "loss_curve.png")
    return loss_history


def save_checkpoint(path, model, optimizer, schedule, epoch, loss_history, config):
    """Save model/optimizer state and training metadata."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device="cpu") -> Dict:
    """Load weights (and optionally optimizer state); returns the checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def plot_loss_curve(loss_history, save_path):
    """Save training loss vs epoch."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
