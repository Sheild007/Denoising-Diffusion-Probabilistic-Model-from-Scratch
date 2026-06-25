from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    """
    Map integer timesteps t to a vector of size embedding_dim.
  """
    timesteps = timesteps.to(torch.float32)
    half_dim = embedding_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim)
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock(nn.Module):
    """
    Basic Residual Block with time embedding injection.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.relu(h)
        h = self.conv2(h)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        return h + self.skip(x)
        

class DownBlock(nn.Module):
    """
    Encoder block with two residual blocks and a max pool layer.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x, time_emb):
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        return x, self.pool(x)


class UpBlock(nn.Module):
    """
    Decoder block: upsample + concat skip + ResidualBlock(s).
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(self, x, skip, time_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        return x
        
class SimpleUNet(nn.Module):
    """
    Lightweight U-Net for noise prediction.

    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256, image_size=64):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, time_emb_dim)
        self.encoder = nn.ModuleList([DownBlock(base_channels, base_channels * 2, time_emb_dim) for _ in range(3)])
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.decoder = nn.ModuleList([UpBlock(base_channels * 4, base_channels * 2, time_emb_dim) for _ in range(3)])
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    def forward(self, x, t):
        t_emb = sinusoidal_timestep_embedding(t, self.time_mlp.in_features)
        t_emb = self.time_mlp(t_emb)
        skips = []
        for down in self.encoder:
            x, skip = down(x, t_emb)
            skips.append(skip)
        x = self.bottleneck(x, t_emb)
        for up, skip in zip(self.decoder, skips[::-1]):
            x = up(x, skip, t_emb)
        return self.out_conv(x) 

def count_parameters(model) -> int:
    """
    Log trainable param count for Report.pdf
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
