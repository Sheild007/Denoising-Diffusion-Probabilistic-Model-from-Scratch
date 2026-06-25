"""
Usage:
  python BSCS22008_05.py train  --data_dir /path/to/animal_data
  python BSCS22008_05.py sample --checkpoint saved_models/diffusion_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from dataset import create_dataloader
from schedule import DiffusionSchedule
from model import SimpleUNet, count_parameters
from train import train, set_seed, load_checkpoint
from sample import sample, save_image_grid


TIMESTEPS = 1000
BASE_CHANNELS = 128
TIME_EMB_DIM = 256
EPOCHS = 1000
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
IMAGE_SIZE = 64


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
    model = SimpleUNet(base_channels=BASE_CHANNELS, time_emb_dim=TIME_EMB_DIM,
                       image_size=args.image_size).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train(
        model, dataloader, schedule, optimizer, device, args.epochs,
        save_dir=args.save_dir, results_dir=args.results_dir,
        loss_type=args.loss_type, config=vars(args),
        grad_clip=GRAD_CLIP, scheduler=scheduler,
    )
    print("Training complete. Checkpoint saved.")


def run_sample(args):
    device = get_device(args.device)
    schedule = DiffusionSchedule(args.timesteps, device=device)
    model = SimpleUNet(base_channels=BASE_CHANNELS, time_emb_dim=TIME_EMB_DIM,
                       image_size=args.image_size).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    images = sample(model, schedule, args.image_size, args.num_samples, device=device)
    out_path = Path(args.results_dir) / "generated_samples.png"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
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
