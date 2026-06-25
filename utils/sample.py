from typing import List, Optional, Tuple

import torch


@torch.no_grad() 
def predict_noise(model, x_t, t, device):
    """
    predict the noise from the model
    """
    t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
    return model(x_t, t_batch)


def compute_denoised_mean(
    x_t:torch.Tensor,
    t: int,
    noise_pred: torch.Tensor,
    schedule,
) -> torch.Tensor:
    """
    Compute the denoised mean from the predicted noise
    """
    alpha_t = schedule.alphas[t]
    alpha_bar_t = schedule.alpha_bar[t]
    coef1 = 1.0 / torch.sqrt(alpha_t)
    coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
    mu = coef1 * (x_t - coef2 * noise_pred)
    return mu


def p_sample(
    model,
    x_t: "torch.Tensor",
    t: int,
    schedule,
    device,
    add_noise: bool = True,
) -> "torch.Tensor":
    """
    implementing Eq. 125 + posterior variance
    """
    noise_pred = predict_noise(model, x_t, t, device)
    mu = compute_denoised_mean(x_t, t, noise_pred, schedule)
    if t > 0 and add_noise:
        z = torch.randn_like(x_t)
        var = schedule.get_posterior_variance(t)
        x_prev = mu + torch.sqrt(var) * z
    else:
        x_prev = mu
    return x_prev


@torch.no_grad() 
def sample(
    model,
    schedule,
    image_size: int,
    batch_size: int = 1,
    channels: int = 3,
    device="cpu",
    show_progress: bool = True,
) -> "torch.Tensor":
    """
    full reverse loop: pure noise = generated images from the model
    """
    model.eval()
    x = torch.randn(batch_size, channels, image_size, image_size, device=device)
    for t in reversed(range(schedule.timesteps)):
        x = p_sample(model, x, t, schedule, device)
    return x

