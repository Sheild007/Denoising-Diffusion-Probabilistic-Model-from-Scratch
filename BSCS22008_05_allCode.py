import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image



SELECTED_CLASSES: List[str] = ["Cat", "Dog", "Bird", "Lion", "Zebra"]
IMAGES_PER_CLASS: int = 80
DEFAULT_IMAGE_SIZE: int = 64
USE_AUGMENTATION: bool = True


def build_transforms(image_size: int = DEFAULT_IMAGE_SIZE, augment: bool = USE_AUGMENTATION) -> Callable:
    ops = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(ops)


def _list_images(folder: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
    paths: List[Path] = []
    for ext in exts:
        paths.extend(folder.glob(ext))
    return sorted(paths, key=lambda p: p.name.lower())


def collect_image_paths(data_dir: str, classes: List[str], images_per_class: Optional[int] = None) -> List[Path]:
    paths: List[Path] = []
    for cls_name in classes:
        cls_path = Path(data_dir) / cls_name
        if not cls_path.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_path}")
        img_paths = _list_images(cls_path)
        if not img_paths:
            raise ValueError(f"No images in {cls_path}")
        if images_per_class is not None:
            if len(img_paths) < images_per_class:
                raise ValueError(f"{cls_name}: need {images_per_class}, found {len(img_paths)}")
            img_paths = img_paths[:images_per_class]
        paths.extend(img_paths)
    return paths


class AnimalDiffusionDataset(Dataset):
    def __init__(self, paths: List[Path], transform: Callable):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def create_dataloader(data_dir, batch_size=16, image_size=DEFAULT_IMAGE_SIZE, shuffle=True,
                      num_workers=2, classes=None, images_per_class=None, augment=USE_AUGMENTATION):
    classes = classes or SELECTED_CLASSES
    images_per_class = IMAGES_PER_CLASS if images_per_class is None else images_per_class
    paths = collect_image_paths(data_dir, classes, images_per_class)
    dataset = AnimalDiffusionDataset(paths, build_transforms(image_size, augment))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())

DEFAULT_T: int = 1000
DEFAULT_BETA_START: float = 1e-4
DEFAULT_BETA_END: float = 0.02


def make_linear_beta_schedule(timesteps=DEFAULT_T, beta_start=DEFAULT_BETA_START, beta_end=DEFAULT_BETA_END):
    return torch.linspace(beta_start, beta_end, timesteps)


def compute_alpha_terms(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
    return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar, "alpha_bar_prev": alpha_bar_prev}


class DiffusionSchedule:
    def __init__(self, timesteps=DEFAULT_T, beta_start=DEFAULT_BETA_START, beta_end=DEFAULT_BETA_END, device="cpu"):
        self.device = torch.device(device)
        self.timesteps = timesteps
        terms = compute_alpha_terms(make_linear_beta_schedule(timesteps, beta_start, beta_end))
        self.betas = terms["betas"].to(self.device)
        self.alphas = terms["alphas"].to(self.device)
        self.alpha_bar = terms["alpha_bar"].to(self.device)
        self.alpha_bar_prev = terms["alpha_bar_prev"].to(self.device)
        self.posterior_variance = torch.clamp(
            (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar) * self.betas, min=1e-20
        )

    def _gather(self, values, t):
        t = t.long().to(values.device)
        return values.gather(0, t).view(-1, 1, 1, 1)

    def get_sqrt_alpha_bar(self, t):
        return torch.sqrt(self._gather(self.alpha_bar, t))

    def get_sqrt_one_minus_alpha_bar(self, t):
        return torch.sqrt(self._gather(1.0 - self.alpha_bar, t))

    def get_posterior_variance(self, t):
        if isinstance(t, int):
            return self.posterior_variance[t]
        t = t.long().to(self.posterior_variance.device)
        return self.posterior_variance.gather(0, t)


def sample_timesteps(batch_size, timesteps, device):
    return torch.randint(0, timesteps, (batch_size,), device=device)


def sample_noise_like(x0):
    return torch.randn_like(x0)


def q_sample(x0, t, noise, schedule):
    sqrt_ab = schedule.get_sqrt_alpha_bar(t)
    sqrt_omab = schedule.get_sqrt_one_minus_alpha_bar(t)
    return sqrt_ab * x0 + sqrt_omab * noise


def forward_diffusion_pair(x0, schedule):
    t = sample_timesteps(x0.shape[0], schedule.timesteps, x0.device)
    noise = sample_noise_like(x0)
    return q_sample(x0, t, noise, schedule), t, noise


def sinusoidal_timestep_embedding(timesteps, embedding_dim):
    timesteps = timesteps.to(torch.float32)
    half_dim = embedding_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim)
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _groups(channels):
    g = min(32, channels)
    while channels % g != 0:
        g -= 1
    return g


class ResidualBlock(nn.Module):
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
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        skip = self.res(x, t_emb)
        return skip, self.pool(skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.res = ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = torch.cat([self.up(x), skip], dim=1)
        return self.res(x, t_emb)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=128, time_emb_dim=256, image_size=64):
        super().__init__()
        bc = base_channels
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim)
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



LossType = Literal["l1", "l2", "mse"]


def diffusion_noise_loss(noise_true, noise_pred, loss_type: LossType = "l2", reduction="mean"):
    if loss_type == "l1":
        return F.l1_loss(noise_pred, noise_true, reduction=reduction)
    return F.mse_loss(noise_pred, noise_true, reduction=reduction)


def train_one_epoch(model, dataloader, schedule, optimizer, device, loss_type="l2", grad_clip=1.0):
    model.train()
    epoch_losses = []
    for x0 in dataloader:
        x0 = x0.to(device)
        x_t, t, noise = forward_diffusion_pair(x0, schedule)
        loss = diffusion_noise_loss(noise, model(x_t, t), loss_type)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        epoch_losses.append(loss.item())
    return float(np.mean(epoch_losses))


def train(model, dataloader, schedule, optimizer, device, epochs, save_dir, results_dir,
          loss_type="l2", save_every=100, log_every=10, config=None, grad_clip=1.0, scheduler=None):
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
            save_checkpoint(Path(save_dir) / "diffusion_model.pt", model, optimizer, schedule, epoch, loss_history, config)
    save_checkpoint(Path(save_dir) / "diffusion_model.pt", model, optimizer, schedule, epochs, loss_history, config)
    plot_loss_curve(loss_history, Path(results_dir) / "loss_curve.png")
    return loss_history


def save_checkpoint(path, model, optimizer, schedule, epoch, loss_history, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def plot_loss_curve(loss_history, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_noise(model, x_t, t, device):
    t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
    return model(x_t, t_batch)


def _scalar(schedule, name, t, x):
    return getattr(schedule, name)[t].to(x.device).view(1, 1, 1, 1)


def compute_denoised_mean(x_t, t, noise_pred, schedule):
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
    mu = compute_denoised_mean(x_t, t, predict_noise(model, x_t, t, device), schedule)
    if t > 0 and add_noise:
        var = schedule.get_posterior_variance(t).to(x_t.device)
        return mu + torch.sqrt(var) * torch.randn_like(x_t)
    return mu


@torch.no_grad()
def sample(model, schedule, image_size, batch_size=1, channels=3, device="cpu"):
    model.eval()
    x = torch.randn(batch_size, channels, image_size, image_size, device=device)
    for t in reversed(range(schedule.timesteps)):
        x = p_sample(model, x, t, schedule, device)
    return x


@torch.no_grad()
def sample_from_noise(model, schedule, initial_noise, device="cpu"):
    model.eval()
    x = initial_noise.to(device)
    for t in reversed(range(schedule.timesteps)):
        x = p_sample(model, x, t, schedule, device)
    return x


def denormalize_for_display(x):
    return (x.clamp(-1, 1) + 1.0) / 2.0


def save_image_grid(images, save_path, nrow=4):
    save_image(denormalize_for_display(images), save_path, nrow=nrow)


IMAGE_SIZE = 64
TIMESTEPS = 1000
BASE_CHANNELS = 128
TIME_EMB_DIM = 256
EPOCHS = 1000
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Diffusion model (DDPM)")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data_dir", required=True)
    t.add_argument("--epochs", type=int, default=EPOCHS)
    t.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    t.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    t.add_argument("--timesteps", type=int, default=TIMESTEPS)
    t.add_argument("--lr", type=float, default=LR)
    t.add_argument("--loss_type", choices=["l1", "l2"], default="l2")
    t.add_argument("--save_dir", default="saved_models")
    t.add_argument("--results_dir", default="results")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default=None)

    s = sub.add_parser("sample")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--num_samples", type=int, default=4)
    s.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    s.add_argument("--timesteps", type=int, default=TIMESTEPS)
    s.add_argument("--results_dir", default="results")
    s.add_argument("--device", default=None)
    return p.parse_args()


def run_train(args):
    set_seed(args.seed)
    device = get_device(args.device)
    print("Device:", device)
    dataloader = create_dataloader(args.data_dir, args.batch_size, args.image_size)
    print(f"Training on {len(dataloader.dataset)} images")
    schedule = DiffusionSchedule(args.timesteps, device=device)
    model = SimpleUNet(base_channels=BASE_CHANNELS, time_emb_dim=TIME_EMB_DIM, image_size=args.image_size).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train(model, dataloader, schedule, optimizer, device, args.epochs,
          save_dir=args.save_dir, results_dir=args.results_dir, loss_type=args.loss_type,
          config=vars(args), grad_clip=GRAD_CLIP, scheduler=scheduler)
    print("Training complete. Checkpoint saved.")


def run_sample(args):
    device = get_device(args.device)
    schedule = DiffusionSchedule(args.timesteps, device=device)
    model = SimpleUNet(base_channels=BASE_CHANNELS, time_emb_dim=TIME_EMB_DIM, image_size=args.image_size).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    images = sample(model, schedule, args.image_size, args.num_samples, device=device)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.results_dir) / "generated_samples.png"
    save_image_grid(images, out_path, nrow=2)
    print("Saved:", out_path)


def main():
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "sample":
        run_sample(args)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
