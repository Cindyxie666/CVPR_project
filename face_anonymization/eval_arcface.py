"""Evaluate a trained ArcFace backbone on the held-out test set.

Reports:
1. Top-1 / Top-5 classification accuracy (nearest-centroid, test-only).
2. Verification metrics: same-identity vs different-identity cosine
   similarity distributions.

Uses the split indices saved by ``train_arcface.py`` to ensure the
evaluation is strictly on images the model never saw during training.

Usage
-----
    python eval_arcface.py
    python eval_arcface.py --weights pretrained/arcface_r50.pth
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models.iresnet import iresnet50


DATA_DIR = "./pins_face/105_classes_pins_dataset"
SPLIT_PATH = "./pretrained/split_indices.pt"
IMAGE_SIZE = 112
BATCH_SIZE = 64


def _build_test_loader(
    data_dir: str, split_path: str,
) -> tuple[DataLoader, list[str]]:
    """Build a DataLoader for the held-out test set only."""
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    full_ds = datasets.ImageFolder(data_dir, transform=tf)

    if os.path.isfile(split_path):
        split = torch.load(split_path, map_location="cpu", weights_only=True)
        test_indices = split["test_indices"]
        test_ds = Subset(full_ds, test_indices)
        print(f"Loaded split from {split_path}  →  test set: {len(test_indices)} images")
    else:
        print(
            f"WARNING: {split_path} not found — evaluating on full dataset. "
            "Run train_arcface.py first to create a proper train/val/test split."
        )
        test_ds = full_ds

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return loader, full_ds.classes


@torch.no_grad()
def extract_all(
    backbone: torch.nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (embeddings, labels) for every image in the loader."""
    all_emb, all_lbl = [], []
    for imgs, labels in tqdm(loader, desc="Extracting embeddings"):
        emb = backbone(imgs.to(device)).cpu()
        all_emb.append(emb)
        all_lbl.append(labels)
    return torch.cat(all_emb), torch.cat(all_lbl)


def classification_accuracy(
    embeddings: torch.Tensor, labels: torch.Tensor, topk: tuple[int, ...] = (1, 5),
) -> dict[int, float]:
    """Nearest-centroid classification accuracy."""
    classes = labels.unique()
    centroids = torch.stack([
        F.normalize(embeddings[labels == c].mean(dim=0), p=2, dim=0)
        for c in classes
    ])

    sims = embeddings @ centroids.T  # (N, C)
    results: dict[int, float] = {}
    for k in topk:
        topk_preds = sims.topk(k, dim=1).indices
        correct = sum(
            labels[i].item() in [classes[p].item() for p in topk_preds[i]]
            for i in range(len(labels))
        )
        results[k] = 100.0 * correct / len(labels)
    return results


def verification_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_pairs: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate positive/negative pairs and compute cosine similarities."""
    class_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels.tolist()):
        class_to_idx[lbl].append(i)
    classes = [c for c in class_to_idx if len(class_to_idx[c]) >= 2]

    pos_sims, neg_sims = [], []
    rng = random.Random(42)

    for _ in range(n_pairs):
        # Positive pair: same class
        c = rng.choice(classes)
        i, j = rng.sample(class_to_idx[c], 2)
        pos_sims.append(F.cosine_similarity(
            embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0),
        ).item())

        # Negative pair: different classes
        c1, c2 = rng.sample(classes, 2)
        i = rng.choice(class_to_idx[c1])
        j = rng.choice(class_to_idx[c2])
        neg_sims.append(F.cosine_similarity(
            embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0),
        ).item())

    return np.array(pos_sims), np.array(neg_sims)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ArcFace backbone")
    parser.add_argument("--weights", type=str, default="./pretrained/arcface_r50.pth")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--split", type=str, default=SPLIT_PATH)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    backbone = iresnet50().to(device)
    if os.path.isfile(args.weights):
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state)
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"WARNING: {args.weights} not found, using random weights")
    backbone.eval()

    loader, class_names = _build_test_loader(args.data_dir, args.split)
    embeddings, labels = extract_all(backbone, loader, device)
    print(f"Extracted {len(embeddings)} embeddings for {len(class_names)} classes")

    # 1. Classification accuracy
    accs = classification_accuracy(embeddings, labels)
    print(f"\nNearest-centroid Top-1 accuracy: {accs[1]:.2f}%")
    print(f"Nearest-centroid Top-5 accuracy: {accs[5]:.2f}%")

    # 2. Verification
    pos_sims, neg_sims = verification_metrics(embeddings, labels)

    best_acc, best_th = 0.0, 0.0
    for th in np.arange(-0.5, 1.0, 0.01):
        tp = (pos_sims >= th).sum()
        tn = (neg_sims < th).sum()
        acc = (tp + tn) / (len(pos_sims) + len(neg_sims))
        if acc > best_acc:
            best_acc, best_th = acc, th

    print(f"\nVerification best accuracy: {best_acc * 100:.2f}%  (threshold={best_th:.2f})")
    print(f"  Positive-pair mean cosine: {pos_sims.mean():.4f} +/- {pos_sims.std():.4f}")
    print(f"  Negative-pair mean cosine: {neg_sims.mean():.4f} +/- {neg_sims.std():.4f}")

    # Plot histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pos_sims, bins=50, alpha=0.6, label="Same identity", density=True)
    ax.hist(neg_sims, bins=50, alpha=0.6, label="Different identity", density=True)
    ax.axvline(best_th, color="red", linestyle="--",
               label=f"Threshold={best_th:.2f} (acc={best_acc*100:.1f}%)")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("ArcFace Verification (Test Set Only)")
    ax.legend()

    hist_path = os.path.join(args.output_dir, "arcface_verification_hist.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHistogram saved -> {hist_path}")


if __name__ == "__main__":
    main()
