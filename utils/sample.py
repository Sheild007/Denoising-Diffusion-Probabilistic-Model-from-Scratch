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

