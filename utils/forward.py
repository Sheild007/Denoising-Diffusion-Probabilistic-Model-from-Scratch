from typing import Optional, Tuple
import torch
from schedule import DiffusionSchedule

def sample_timesteps(
    batch_size: int,
    timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Pick random t from Uniform({0, 1, ..., T-1}) for each batch item.
    """
    return torch.randint(0, timesteps, (batch_size,), device=device)


def sample_noise_like(x0: torch.Tensor) -> torch.Tensor:
    """
    Randomly sample noise from N(0, I) with the same shape as x0.
    """
    return torch.randn_like(x0)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """
    Implement Eq. 70:
      sqrt_ab          = schedule.get_sqrt_alpha_bar(t)   
      sqrt_one_minus   = schedule.get_sqrt_one_minus_alpha_bar(t)
      x_t = sqrt_ab * x0 + sqrt_one_minus * noise
      return x_t

    """
    return torch.sqrt(schedule.get_sqrt_alpha_bar(t)) * x0 + torch.sqrt(schedule.get_sqrt_one_minus_alpha_bar(t)) * noise


def forward_diffusion_pair(
    x0:torch.Tensor,
    schedule:DiffusionSchedule ,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    implement the forward diffusion process
    """
    t = sample_timesteps(x0.shape[0], schedule.timesteps, x0.device)
    noise = sample_noise_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    return x_t, t, noise

