"""Fine-tune an ArcFace model on the Pins Face Recognition dataset.

Usage
-----
    python train_arcface.py --pretrained ../backbone.pth
    python train_arcface.py --pretrained pretrained/arcface_r50.pth --epochs 15

Loads a pretrained IResNet-50 backbone (e.g. from insightface model zoo)
and trains a cosine classifier head on 105 celebrity identities.

Phase 1 (first half): backbone frozen, only head trains.
Phase 2 (second half): backbone unfrozen with very low LR.

Output
------
    pretrained/arcface_r50.pth   — backbone state dict (512-d embeddings)
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from models.iresnet import iresnet50


# ─────────────── Defaults ───────────────

DATA_DIR = "./pins_face/105_classes_pins_dataset"
OUTPUT_PATH = "pretrained/arcface_r50.pth"
IMAGE_SIZE = 112
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

# Scale for cosine logits (no angular margin — just scaled cosine softmax)
LOGIT_SCALE = 16.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────── Cosine Classifier Head ───────────────


class CosineClassifier(nn.Module):
    """Normalised-weight linear layer → scaled cosine similarity logits.

    No angular margin — just ``s * cos(θ)`` followed by cross-entropy.
    This is the correct approach for fine-tuning a pretrained backbone
    on a small downstream dataset.
    """

    def __init__(self, embedding_dim: int = 512, num_classes: int = 105,
                 scale: float = 16.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, w)       # (B, C) in [-1, 1]
        return cosine * self.scale


# ─────────────── Data ───────────────


def build_loaders(
    data_dir: str, image_size: int, batch_size: int,
    val_ratio: float = 0.1, test_ratio: float = 0.15,
) -> tuple[DataLoader, DataLoader, int]:
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    full_ds = datasets.ImageFolder(data_dir, transform=train_tf)
    num_classes = len(full_ds.classes)
    n_total = len(full_ds)

    test_size = max(1, int(n_total * test_ratio))
    val_size = max(1, int(n_total * val_ratio))
    train_size = n_total - val_size - test_size
    train_ds, val_ds, test_ds = random_split(
        full_ds, [train_size, val_size, test_size],
    )

    split_path = os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "split_indices.pt")
    os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
    torch.save({
        "train_indices": train_ds.indices,
        "val_indices": val_ds.indices,
        "test_indices": test_ds.indices,
    }, split_path)
    print(f"Classes: {num_classes}  |  "
          f"Train: {train_size}  |  Val: {val_size}  |  Test: {test_size}")
    print(f"Split indices saved to {split_path}")

    val_ds.dataset = datasets.ImageFolder(data_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader, num_classes


# ─────────────── Training ───────────────


def train(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, num_classes = build_loaders(
        args.data_dir, IMAGE_SIZE, args.batch_size,
    )

    # ── Build backbone ──
    backbone = iresnet50(dropout=0.0).to(device)

    if args.pretrained and os.path.isfile(args.pretrained):
        print(f"[Backbone] Loading pretrained weights from {args.pretrained}")
        state = torch.load(args.pretrained, map_location=device, weights_only=True)
        backbone.load_state_dict(state, strict=True)
        print("[Backbone] Loaded successfully")
    else:
        print("[Backbone] WARNING: No pretrained weights — training from scratch")

    # ── Cosine classifier head (no angular margin) ──
    head = CosineClassifier(512, num_classes, scale=LOGIT_SCALE).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase boundary: first half = frozen backbone, second half = unfrozen
    unfreeze_epoch = args.epochs // 2 + 1

    # Start with backbone frozen
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"[Phase 1] Epochs 1–{unfreeze_epoch - 1}: backbone FROZEN, training head only")

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Phase transition: unfreeze backbone ──
        if epoch == unfreeze_epoch:
            print(f"\n[Phase 2] Epochs {unfreeze_epoch}–{args.epochs}: "
                  f"backbone UNFROZEN (lr=1e-5)")
            for p in backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": backbone.parameters(), "lr": 1e-5},
                {"params": head.parameters(), "lr": 3e-4},
            ])

        # ── Train ──
        if epoch >= unfreeze_epoch:
            backbone.train()
        head.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            if epoch < unfreeze_epoch:
                with torch.no_grad():
                    embeddings = backbone(images)
            else:
                embeddings = backbone(images)

            logits = head(embeddings)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

            if total % (args.batch_size * 20) == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100.0 * correct / total:.1f}%",
                )

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        # ── Validate ──
        backbone.eval()
        head.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                embeddings = backbone(images)
                logits = head(embeddings)
                v_correct += logits.argmax(1).eq(labels).sum().item()
                v_total += labels.size(0)
        val_acc = 100.0 * v_correct / v_total

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(backbone.state_dict(), args.output)
            tag = "  ** BEST **"

        phase = "frozen" if epoch < unfreeze_epoch else "unfrozen"
        print(
            f"[Epoch {epoch}] ({phase})  "
            f"loss: {train_loss:.4f}  train_acc: {train_acc:.1f}%  "
            f"val_acc: {val_acc:.1f}%{tag}"
        )

    print(f"\nBest val accuracy: {best_val_acc:.2f}%")
    print(f"Backbone weights saved to: {args.output}")


# ─────────────── CLI ───────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune ArcFace on Pins Face Recognition dataset",
    )
    parser.add_argument("--pretrained", type=str, default="pretrained/pretrained_r50.pth",
                        help="Path to pretrained IResNet-50 backbone.pth")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
