from typing import Literal

import torch
import torch.nn.functional as F

LossType = Literal["l1", "l2", "mse"]

def diffusion_noise_loss(
    noise_true: torch.Tensor,
    noise_pred: torch.Tensor,
    loss_type: LossType = "l2",
    reduction: str = "mean",
) -> torch.Tensor:
    """Compare ground-truth noise with predicted noise (Eq. 130)."""
    if loss_type == "l1":
        return F.l1_loss(noise_pred, noise_true, reduction=reduction)
    return F.mse_loss(noise_pred, noise_true, reduction=reduction)


def diffusion_noise_loss_per_sample(
    noise_true: torch.Tensor,
    noise_pred: torch.Tensor,
    loss_type: LossType = "l2",
) -> torch.Tensor:
    """Per-image loss [B] for logging the hardest samples."""
    if loss_type == "l1":
        per_pixel = (noise_pred - noise_true).abs()
    else:
        per_pixel = (noise_pred - noise_true) ** 2
    return per_pixel.flatten(1).mean(dim=1)
