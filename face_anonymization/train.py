"""Training script for the face anonymization U-Net.

Usage
-----
    python train.py                          # use defaults from config.py
    python train.py --epochs 50 --batch_size 16 --device cuda

The script:
1. Loads frozen ArcFace and expression models (or ImageNet placeholders).
2. Builds the U-Net anonymizer.
3. Trains with the composite loss (identity + expression + recon + perceptual).
4. Saves checkpoints and sample visualisations periodically.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models as tv_models
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from dataset import build_dataloaders
from models.unet import UNetAnonymizer
from models.iresnet import iresnet50
from models.losses import AnonymizationLoss


# ─────────────── Model loaders ───────────────


def _load_arcface(cfg: Config, device: torch.device) -> nn.Module:
    """Load the ArcFace (IResNet-50) face recognition backbone.

    Falls back to a ResNet-50 embedding model if no pretrained weights are
    found — useful for development before the teammate's model is ready.
    """
    model = iresnet50()

    if os.path.isfile(cfg.arcface_weights):
        print(f"[ArcFace] loading weights from {cfg.arcface_weights}")
        state = torch.load(cfg.arcface_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    else:
        print(
            f"[ArcFace] WARNING  {cfg.arcface_weights} not found. "
            "Using randomly initialised IResNet-50 as placeholder. "
            "Training will still run but identity loss will be weak."
        )

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_expression_model(cfg: Config, device: torch.device) -> nn.Module:
    """Load the ResNet-18 expression classifier (teammate's model).

    Architecture matches the FER-2013 trainer:
        resnet18 → fc = Dropout→Linear(512,128)→ReLU→Dropout→Linear(128, N)
    """
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, cfg.expression_num_classes),
    )

    if os.path.isfile(cfg.expression_weights):
        print(f"[Expression] loading weights from {cfg.expression_weights}")
        state = torch.load(
            cfg.expression_weights, map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state, strict=False)
    else:
        print(
            f"[Expression] WARNING  {cfg.expression_weights} not found. "
            "Using ImageNet-pretrained ResNet-18 as placeholder."
        )

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ─────────────── Visualisation helper ───────────────


def _save_samples(
    G: nn.Module,
    val_loader,
    epoch: int,
    device: torch.device,
    out_dir: str,
) -> None:
    """Save a grid of original / anonymised image pairs."""
    G.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))[:8].to(device)
        anon = G(batch)
        # Interleave: orig1, anon1, orig2, anon2, ...
        pairs = torch.stack([batch, anon], dim=1).view(-1, *batch.shape[1:])
        save_image(
            pairs * 0.5 + 0.5,  # back to [0,1]
            os.path.join(out_dir, f"samples_epoch{epoch:04d}.png"),
            nrow=4,
            padding=2,
        )
    G.train()


# ─────────────── Training loop ───────────────


def train(cfg: Config) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Data
    train_loader, val_loader = build_dataloaders(
        cfg.data_dir, cfg.image_size, cfg.batch_size, cfg.num_workers,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Models
    G = UNetAnonymizer(base_ch=cfg.base_channels).to(device)
    arcface = _load_arcface(cfg, device)
    expr_model = _load_expression_model(cfg, device)

    criterion = AnonymizationLoss(arcface, expr_model, cfg).to(device)

    # Optimiser (only G's parameters)
    optimizer = Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    # ── Resume from checkpoint ──
    start_epoch = 1
    if cfg.resume:
        if not os.path.isfile(cfg.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            G.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from {cfg.resume}  (epoch {ckpt['epoch']}, "
                  f"continuing from epoch {start_epoch})")
        else:
            G.load_state_dict(ckpt)
            print(f"Loaded model weights from {cfg.resume}  "
                  f"(old format — optimizer/scheduler reset, starting epoch 1)")

    # ── Training ──
    for epoch in range(start_epoch, cfg.epochs + 1):
        G.train()
        epoch_losses: dict[str, float] = {}
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)
            anonymized = G(images)

            losses = criterion(images, anonymized)

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=10.0)
            optimizer.step()

            # Accumulate for logging
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            if batch_idx % 50 == 0:
                pbar.set_postfix(
                    total=f"{losses['total'].item():.4f}",
                    iden=f"{losses['identity'].item():.3f}",
                    expr=f"{losses['expression'].item():.3f}",
                )

        scheduler.step()

        # Average losses
        n = len(train_loader)
        summary = " | ".join(f"{k}: {v / n:.4f}" for k, v in epoch_losses.items())
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] {summary}  ({elapsed:.1f}s)")

        # Save sample images
        if epoch % cfg.sample_every == 0:
            _save_samples(G, val_loader, epoch, device, cfg.output_dir)

        # Checkpoint
        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            path = os.path.join(cfg.checkpoint_dir, f"anonymizer_epoch{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "model": G.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, path)
            print(f"  checkpoint saved → {path}")

    print("Training complete.")


# ─────────────── CLI ───────────────


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train face anonymization U-Net")
    for field_name, field_val in vars(cfg).items():
        ftype = type(field_val)
        if ftype is tuple:
            continue
        parser.add_argument(f"--{field_name}", type=ftype, default=field_val)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    train(parse_args())
