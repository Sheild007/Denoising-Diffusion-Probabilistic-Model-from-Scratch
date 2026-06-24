from typing import Dict, Union
import matplotlib.pyplot as plt
import torch

DEFAULT_T: int = 1000
DEFAULT_BETA_START: float = 1e-4
DEFAULT_BETA_END: float = 0.02


def make_linear_beta_schedule(
    timesteps: int = DEFAULT_T,
    beta_start: float = DEFAULT_BETA_START,
    beta_end: float = DEFAULT_BETA_END,
) -> torch.Tensor:
    """Linear β schedule from beta_start to beta_end over `timesteps` steps."""
    return torch.linspace(beta_start, beta_end, timesteps)


def compute_alpha_terms(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas,dim=0)
    alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
    }


class DiffusionSchedule:
    """
    Precomputed diffusion schedule tensors on a chosen device.

    """

    def __init__(
        self,
        timesteps: int = DEFAULT_T,
        beta_start: float = DEFAULT_BETA_START,
        beta_end: float = DEFAULT_BETA_END,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.timesteps = timesteps

        betas = make_linear_beta_schedule(timesteps, beta_start, beta_end)
        terms = compute_alpha_terms(betas)

        self.betas = terms["betas"].to(self.device)
        self.alphas = terms["alphas"].to(self.device)
        self.alpha_bar = terms["alpha_bar"].to(self.device)
        self.alpha_bar_prev = terms["alpha_bar_prev"].to(self.device)

        # Posterior variance for reverse sampling 
        self.posterior_variance = (
            (1.0 - self.alpha_bar_prev)
            / (1.0 - self.alpha_bar)
            * self.betas
        )
        # At t=0 denominator can be tiny; clamp for numerical stability
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def _gather(self, values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.long().to(values.device)
        out = values.gather(0, t)
        return out.view(-1, 1, 1, 1)

    def get_sqrt_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """sqrt(alpha_bar) for Eq. 70 forward sampling."""
        return torch.sqrt(self._gather(self.alpha_bar, t))

    def get_sqrt_one_minus_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """sqrt(1 - alpha_bar) for Eq. 70 forward sampling."""
        return torch.sqrt(self._gather(1.0 - self.alpha_bar, t))

    def get_posterior_variance(
        self, t: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """posterior variance for reverse sampling"""
        if isinstance(t, int):
            return self.posterior_variance[t]
        t = t.long().to(self.posterior_variance.device)
        return self.posterior_variance.gather(0, t)


def verify_schedule(
    schedule: DiffusionSchedule,
    save_path: str | None = None,
) -> None:
    
    assert schedule.alpha_bar[0] > 0.99, "Early alpha_bar should be near 1"
    assert schedule.alpha_bar[-1] < 0.01, "Final alpha_bar should be near 0 (pure noise)"

    if save_path is None:
        return

    steps = torch.arange(schedule.timesteps)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(steps.numpy(), schedule.betas.cpu().numpy())
    axes[0].set_title("beta (noise per step)")
    axes[0].set_xlabel("t")

    axes[1].plot(steps.numpy(), schedule.alpha_bar.cpu().numpy())
    axes[1].set_title("alpha_bar (signal remaining)")
    axes[1].set_xlabel("t")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
