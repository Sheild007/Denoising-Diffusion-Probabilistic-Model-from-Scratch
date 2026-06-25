from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Assignment: 5 classes (minimum 4 images each; use more on Kaggle)
SELECTED_CLASSES: List[str] = [
    "Cat",
    "Dog",
    "Bird",
    "Lion",
    "Zebra",
]

IMAGES_PER_CLASS: int = 80   # 5 × 80 = 400 images
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


def collect_image_paths(
    data_dir: str,
    classes: List[str],
    images_per_class: Optional[int] = None,
) -> List[Path]:
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
                raise ValueError(
                    f"{cls_name}: need {images_per_class}, found {len(img_paths)}"
                )
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


def create_dataloader(
    data_dir: str,
    batch_size: int = 16,
    image_size: int = DEFAULT_IMAGE_SIZE,
    shuffle: bool = True,
    num_workers: int = 2,
    classes: Optional[List[str]] = None,
    images_per_class: Optional[int] = None,
    augment: bool = USE_AUGMENTATION,
):
    classes = classes or SELECTED_CLASSES
    images_per_class = IMAGES_PER_CLASS if images_per_class is None else images_per_class
    paths = collect_image_paths(data_dir, classes, images_per_class)
    dataset = AnimalDiffusionDataset(paths, build_transforms(image_size, augment))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
