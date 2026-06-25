from typing import Dict, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

from forward import forward_diffusion_pair
from loss import diffusion_noise_loss

def train_one_epoch(
    model,
    dataloader,
    schedule,
    optimizer,
    device,
    loss_type: str = "l2",
) -> float:
    
    epoch_losses = []
    for x0 in dataloader:
        x0 = x0.to(device)
        x_t, t, noise = forward_diffusion_pair(x0, schedule)
        noise_pred = model(x_t, t)
        loss = diffusion_noise_loss(noise, noise_pred, loss_type)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        epoch_losses.append(loss.item())
    return float(np.mean(epoch_losses))

