"""
model.py - U-Net that predicts the added noise epsilon_theta(x_t, t).

Timestep is injected via a sinusoidal embedding (Eq. 125-130 use this
noise-prediction parameterization).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Map integer timesteps to a vector (Transformer-style positional encoding)."""
    timesteps = timesteps.to(torch.float32)
    half_dim = embedding_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim)
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _groups(channels: int) -> int:
    g = min(32, channels)
    while channels % g != 0:
        g -= 1
    return g


class ResidualBlock(nn.Module):
    """Conv -> norm -> SiLU with timestep embedding added, plus residual skip."""

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.drop(self.act(self.norm2(h))))
        return h + self.skip(x)


class DownBlock(nn.Module):
    """Residual block then 2x downsample; returns the pre-pool skip and the pooled output."""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        skip = self.res(x, t_emb)
        return skip, self.pool(skip)


class UpBlock(nn.Module):
    """2x upsample, concat skip, then a residual block."""

    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.res = ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


class SimpleUNet(nn.Module):
    """U-Net (64 -> 32 -> 16 -> 8 -> 64) predicting noise of shape [B, 3, H, W]."""

    def __init__(self, in_channels=3, out_channels=3, base_channels=128, time_emb_dim=256, image_size=64):
        super().__init__()
        bc = base_channels
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, bc, 3, padding=1)
        self.down1 = DownBlock(bc, bc, time_emb_dim)
        self.down2 = DownBlock(bc, bc * 2, time_emb_dim)
        self.down3 = DownBlock(bc * 2, bc * 4, time_emb_dim)
        self.bottleneck = ResidualBlock(bc * 4, bc * 4, time_emb_dim)
        self.up1 = UpBlock(bc * 4, bc * 4, bc * 2, time_emb_dim)
        self.up2 = UpBlock(bc * 2, bc * 2, bc, time_emb_dim)
        self.up3 = UpBlock(bc, bc, bc, time_emb_dim)
        self.out_conv = nn.Conv2d(bc, out_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_emb_dim))
        x = self.in_conv(x)
        s1, x = self.down1(x, t_emb)
        s2, x = self.down2(x, t_emb)
        s3, x = self.down3(x, t_emb)
        x = self.bottleneck(x, t_emb)
        x = self.up1(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up3(x, s1, t_emb)
        return self.out_conv(x)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
