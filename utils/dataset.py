from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

SELECTED_CLASSES: List[str] = [
    "Cat",
    "Dog",
    "Bird",
    "Lion",
    "Zebra",
]

IMAGES_PER_CLASS: int = 4
DEFAULT_IMAGE_SIZE: int = 64

def build_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> Callable:
    """
    Resizing image and normalizing to [-1,1]
    """
    return transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5],std=[0.5]),
    ])

