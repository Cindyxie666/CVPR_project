"""CelebA-HQ dataset loader for face anonymization training.

Images are resized to ``image_size × image_size`` and normalised to [-1, 1]
so that the U-Net tanh output lives in the same range.
"""

from __future__ import annotations

import os
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class CelebAHQDataset(Dataset):
    """Load images from a flat directory of .jpg / .png files."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: str, image_size: int = 256):
        self.root_dir = root_dir
        self.paths = sorted(
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No images found in {root_dir}. "
                "Expected .jpg/.png files in a flat directory."
            )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),                # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> "torch.Tensor":
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_dataloaders(
    data_dir: str,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    val_ratio: float = 0.05,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) with a small validation split."""
    full_ds = CelebAHQDataset(data_dir, image_size)
    val_size = max(1, int(len(full_ds) * val_ratio))
    train_size = len(full_ds) - val_size

    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
