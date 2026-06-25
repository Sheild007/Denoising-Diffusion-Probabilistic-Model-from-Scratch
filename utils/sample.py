"""
sample.py - Reverse diffusion (Eq. 125 + DDPM posterior).

Starts from x_T ~ N(0, I) and denoises down to x_0. The reverse step
recovers x_0 from the predicted noise and clips it to [-1, 1] for stability,
then computes the posterior mean q(x_{t-1} | x_t, x_0).
"""

from typing import List

import torch
from torchvision.utils import make_grid, save_image


@torch.no_grad()
def predict_noise(model, x_t, t, device):
    t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
    return model(x_t, t_batch)


def _scalar(schedule, name, t, x):
    return getattr(schedule, name)[t].to(x.device).view(1, 1, 1, 1)


def compute_denoised_mean(x_t, t, noise_pred, schedule):
    """Posterior mean using x0 recovered from predicted noise (clipped)."""
    sqrt_ab = _scalar(schedule, "alpha_bar", t, x_t).sqrt()
    sqrt_omab = (1.0 - _scalar(schedule, "alpha_bar", t, x_t)).sqrt()
    x0_pred = ((x_t - sqrt_omab * noise_pred) / (sqrt_ab + 1e-8)).clamp(-1.0, 1.0)

    beta_t = _scalar(schedule, "betas", t, x_t)
    alpha_t = _scalar(schedule, "alphas", t, x_t)
    alpha_bar_t = _scalar(schedule, "alpha_bar", t, x_t)
    alpha_bar_prev = _scalar(schedule, "alpha_bar_prev", t, x_t)

    coef_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1.0 - alpha_bar_t)
    coef_xt = (alpha_t.sqrt() * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
    return coef_x0 * x0_pred + coef_xt * x_t


def p_sample(model, x_t, t, schedule, device, add_noise=True):
    """Single reverse step x_t -> x_{t-1}."""
    noise_pred = predict_noise(model, x_t, t, device)
    mu = compute_denoised_mean(x_t, t, noise_pred, schedule)
    if t > 0 and add_noise:
        var = schedule.get_posterior_variance(t).to(x_t.device)
        return mu + torch.sqrt(var) * torch.randn_like(x_t)
    return mu


@torch.no_grad()
def sample(model, schedule, image_size, batch_size=1, channels=3, device="cpu"):
    """Full reverse loop: pure noise -> generated images."""
    model.eval()
    x = torch.randn(batch_size, channels, image_size, image_size, device=device)
    for t in reversed(range(schedule.timesteps)):
        x = p_sample(model, x, t, schedule, device)
    return x


@torch.no_grad()
def sample_from_noise(model, schedule, initial_noise, device="cpu"):
    """Reverse loop starting from a given x_T (reproducible demo)."""
    model.eval()
    x = initial_noise.to(device)
    for t in reversed(range(schedule.timesteps)):
        x = p_sample(model, x, t, schedule, device)
    return x


def denormalize_for_display(x):
    return (x.clamp(-1, 1) + 1.0) / 2.0


def save_image_grid(images, save_path, nrow=4):
    save_image(denormalize_for_display(images), save_path, nrow=nrow)


@torch.no_grad()
def visualize_reverse_steps(model, schedule, image_size, steps_to_save, save_path, device="cpu"):
    """Save snapshots of x during the reverse loop (noise -> image)."""
    model.eval()
    steps_to_save = set(steps_to_save)
    x = torch.randn(1, 3, image_size, image_size, device=device)
    frames = []
    for t in reversed(range(schedule.timesteps)):
        x = p_sample(model, x, t, schedule, device)
        if t in steps_to_save:
            frames.append(denormalize_for_display(x[0].cpu()))
    grid = make_grid(torch.stack(frames), nrow=len(frames))
    save_image(grid, save_path)
