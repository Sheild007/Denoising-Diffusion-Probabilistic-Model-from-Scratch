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


def collect_image_paths(
    data_dir: str,
    classes: List[str],
    images_per_class: int,
) -> List[Path]:
    """
    Collecting image paths from the data directory
    """
    paths: List[Path] = []
    for cls_name in classes:
      cls_path = Path(data_dir) / cls_name
      img_paths = sorted(cls_path.glob("*.jpg")) + sorted(cls_path.glob("*.jpeg"))
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
      path = self.paths[index]
      image = Image.open(path).convert("RGB")

      if self.transform:
        image = self.transform(image)

      return image
