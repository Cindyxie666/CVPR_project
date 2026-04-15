"""Evaluate a trained face anonymization model.

Two separate evaluations, each on the proper test dataset:

1. **Privacy test** (Pins Face held-out test set)
   – Loads the train/test split saved by ``train_arcface.py``
     (``split_indices.pt``) so we only evaluate on images ArcFace never saw.
   – Re-identification rate: fraction of anonymised images whose nearest
     gallery centroid is still the correct identity.  *Lower = better.*
   – Mean cosine similarity (orig vs anon).  *Lower = better.*

2. **Utility test** (FER-2013 test set)
   – Expression accuracy on originals vs anonymised images.
   – Expression consistency: fraction where the prediction matches
     between original and anonymised.  *Higher = better.*

Usage
-----
    python evaluate.py --checkpoint checkpoints/anonymizer_epoch0010.pth

    python evaluate.py \
        --checkpoint checkpoints/anonymizer_epoch0010.pth \
        --pins_dir  ../pins_face/105_classes_pins_dataset \
        --fer_dir   ../expression_recognition/expression_recognition/data/test
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models as tv_models, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from models.unet import UNetAnonymizer
from models.iresnet import iresnet50
from models.losses import prep_arcface, prep_expression


# ────────────────────── Default paths ──────────────────────

PINS_DIR = "./pins_face/105_classes_pins_dataset"
FER_DIR = "./expression_recognition/expression_recognition/data/test"
SPLIT_PATH = "./pretrained/split_indices.pt"


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy, macro-precision, macro-recall, macro-F1."""
    acc = (y_true == y_pred).mean() * 100
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    return {
        "accuracy": acc,
        "precision": p * 100,
        "recall": r * 100,
        "f1": f1 * 100,
    }


def _print_metrics(header: str, m: dict[str, float]) -> None:
    print(f"  {header}")
    print(f"    Accuracy  : {m['accuracy']:.1f}%")
    print(f"    Precision : {m['precision']:.1f}%  (macro)")
    print(f"    Recall    : {m['recall']:.1f}%  (macro)")
    print(f"    F1 Score  : {m['f1']:.1f}%  (macro)")


# ────────────────────── Model loaders ──────────────────────


def _load_anonymizer(
    checkpoint: str, cfg: Config, device: torch.device,
) -> nn.Module:
    G = UNetAnonymizer(base_ch=cfg.base_channels).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        G.load_state_dict(ckpt["model"])
    else:
        G.load_state_dict(ckpt)
    G.eval()
    print(f"[Anonymizer] loaded {checkpoint}")
    return G


def _load_arcface(cfg: Config, device: torch.device) -> nn.Module:
    model = iresnet50()
    if os.path.isfile(cfg.arcface_weights):
        state = torch.load(cfg.arcface_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"[ArcFace] loaded {cfg.arcface_weights}")
    else:
        print("[ArcFace] WARNING  using randomly initialised weights")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_expression_model(cfg: Config, device: torch.device) -> nn.Module:
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, cfg.expression_num_classes),
    )
    if os.path.isfile(cfg.expression_weights):
        state = torch.load(
            cfg.expression_weights, map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state, strict=False)
        print(f"[Expression] loaded {cfg.expression_weights}")
    else:
        print("[Expression] WARNING  using ImageNet-pretrained placeholder")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ────────────────────── Test 1: Privacy (Pins Face) ──────────────────────


@torch.no_grad()
def evaluate_privacy(
    G: nn.Module,
    arcface: nn.Module,
    pins_dir: str,
    split_path: str,
    output_dir: str,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Re-identification test on the held-out Pins Face test set."""
    print("\n" + "=" * 60)
    print("TEST 1: PRIVACY — Anonymization + ArcFace on Pins Face")
    print("=" * 60)

    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    full_ds = datasets.ImageFolder(pins_dir, transform=tf)

    if not os.path.isfile(split_path):
        raise FileNotFoundError(
            f"{split_path} not found. Run train_arcface.py first to "
            "generate the train/test split."
        )
    split = torch.load(split_path, map_location="cpu", weights_only=True)
    test_indices = split["test_indices"]
    test_ds = Subset(full_ds, test_indices)
    print(f"Held-out test set: {len(test_indices)} images (from {split_path})")

    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    class_embs: dict[int, list[torch.Tensor]] = {}
    all_cos: list[float] = []
    emb_orig_all: list[torch.Tensor] = []
    emb_anon_all: list[torch.Tensor] = []
    label_all: list[torch.Tensor] = []
    sample_orig, sample_anon = [], []

    for images, labels in tqdm(loader, desc="Privacy eval"):
        images = images.to(device)
        anonymized = G(images)

        emb_o = arcface(prep_arcface(images))
        emb_a = arcface(prep_arcface(anonymized))

        emb_orig_all.append(emb_o.cpu())
        emb_anon_all.append(emb_a.cpu())
        label_all.append(labels)

        cos = F.cosine_similarity(emb_o, emb_a, dim=1)
        all_cos.extend(cos.cpu().tolist())

        for i, lbl in enumerate(labels.tolist()):
            class_embs.setdefault(lbl, []).append(emb_o[i].cpu())

        if len(sample_orig) < 16:
            sample_orig.append(images[:4].cpu())
            sample_anon.append(anonymized[:4].cpu())

    # Gallery: one centroid per identity from original embeddings
    centroids_dict = {
        c: F.normalize(torch.stack(embs).mean(dim=0), p=2, dim=0)
        for c, embs in class_embs.items()
    }
    gallery_classes = sorted(centroids_dict.keys())
    gallery = torch.stack([centroids_dict[c] for c in gallery_classes])

    all_emb_orig = torch.cat(emb_orig_all)
    all_emb_anon = torch.cat(emb_anon_all)
    gt = torch.cat(label_all).numpy()

    # Nearest-centroid classification on originals (baseline)
    sims_orig = all_emb_orig @ gallery.T
    pred_orig = np.array([gallery_classes[i] for i in sims_orig.argmax(dim=1).tolist()])

    # Nearest-centroid classification on anonymised
    sims_anon = all_emb_anon @ gallery.T
    pred_anon = np.array([gallery_classes[i] for i in sims_anon.argmax(dim=1).tolist()])

    orig_metrics = _classification_metrics(gt, pred_orig)
    anon_metrics = _classification_metrics(gt, pred_anon)

    cosine_sims = np.array(all_cos)
    mean_cos = cosine_sims.mean()

    print(f"\n  Images evaluated       : {len(gt)}")
    print(f"  Mean cos(orig, anon)   : {mean_cos:.4f}  (lower = better privacy)")
    print()
    _print_metrics("ArcFace on ORIGINAL faces (baseline):", orig_metrics)
    print()
    _print_metrics("ArcFace on ANONYMISED faces:", anon_metrics)
    print()
    print(f"  Accuracy drop          : "
          f"{orig_metrics['accuracy'] - anon_metrics['accuracy']:.1f}%")
    print(f"  F1 drop                : "
          f"{orig_metrics['f1'] - anon_metrics['f1']:.1f}%")

    # Visualisation
    orig_imgs = torch.cat(sample_orig, dim=0)[:16]
    anon_imgs = torch.cat(sample_anon, dim=0)[:16]
    pairs = torch.stack([orig_imgs, anon_imgs], dim=1).reshape(
        -1, *orig_imgs.shape[1:],
    )
    grid_path = os.path.join(output_dir, "privacy_comparison.png")
    save_image(pairs * 0.5 + 0.5, grid_path, nrow=4, padding=2)
    print(f"  Comparison grid        : {grid_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cosine_sims, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0.5, color="red", linestyle="--", label="Threshold 0.5")
    ax.set_xlabel("Cosine Similarity (orig vs anon)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Privacy: Identity Distance (Pins Face Test Set)\n"
        f"Mean={mean_cos:.3f} | Re-ID={anon_metrics['accuracy']:.1f}%"
    )
    ax.legend()
    hist_path = os.path.join(output_dir, "privacy_cosine_hist.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cosine histogram       : {hist_path}")

    return {
        "orig": orig_metrics, "anon": anon_metrics,
        "mean_cos": mean_cos, "total": len(gt),
    }


# ────────────────────── Test 2: Utility (FER-2013) ──────────────────────


@torch.no_grad()
def evaluate_utility(
    G: nn.Module,
    expr_model: nn.Module,
    fer_dir: str,
    output_dir: str,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Expression preservation test on the FER-2013 test set."""
    print("\n" + "=" * 60)
    print("TEST 2: UTILITY — Anonymization + Expression on FER-2013")
    print("=" * 60)

    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ds = datasets.ImageFolder(fer_dir, transform=tf)
    class_names = ds.classes
    print(f"FER-2013 test set: {len(ds)} images, classes: {class_names}")

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    all_gt: list[int] = []
    all_pred_orig: list[int] = []
    all_pred_anon: list[int] = []
    sample_orig, sample_anon = [], []

    for images, labels in tqdm(loader, desc="Utility eval"):
        images = images.to(device)
        labels = labels.to(device)
        anonymized = G(images)

        pred_orig = expr_model(prep_expression(images)).argmax(dim=1)
        pred_anon = expr_model(prep_expression(anonymized)).argmax(dim=1)

        all_gt.extend(labels.cpu().tolist())
        all_pred_orig.extend(pred_orig.cpu().tolist())
        all_pred_anon.extend(pred_anon.cpu().tolist())

        if len(sample_orig) < 16:
            sample_orig.append(images[:4].cpu())
            sample_anon.append(anonymized[:4].cpu())

    gt = np.array(all_gt)
    po = np.array(all_pred_orig)
    pa = np.array(all_pred_anon)

    orig_metrics = _classification_metrics(gt, po)
    anon_metrics = _classification_metrics(gt, pa)
    consistency = (po == pa).mean() * 100

    print(f"\n  Images evaluated       : {len(gt)}")
    print(f"  Expression classes     : {class_names}")
    print()
    _print_metrics("Expression on ORIGINAL faces (baseline):", orig_metrics)
    print()
    _print_metrics("Expression on ANONYMISED faces:", anon_metrics)
    print()
    print(f"  Accuracy drop          : "
          f"{orig_metrics['accuracy'] - anon_metrics['accuracy']:.1f}%")
    print(f"  F1 drop                : "
          f"{orig_metrics['f1'] - anon_metrics['f1']:.1f}%")
    print(f"  Expression consistency : {consistency:.1f}%  (higher = better)")

    # Visualisation
    orig_imgs = torch.cat(sample_orig, dim=0)[:16]
    anon_imgs = torch.cat(sample_anon, dim=0)[:16]
    pairs = torch.stack([orig_imgs, anon_imgs], dim=1).reshape(
        -1, *orig_imgs.shape[1:],
    )
    grid_path = os.path.join(output_dir, "utility_comparison.png")
    save_image(pairs * 0.5 + 0.5, grid_path, nrow=4, padding=2)
    print(f"  Comparison grid        : {grid_path}")

    return {
        "orig": orig_metrics, "anon": anon_metrics,
        "consistency": consistency, "total": len(gt),
    }


# ────────────────────── CLI ──────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate face anonymization: privacy + utility",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="./checkpoints/anonymizer_epoch0010.pth",
        help="Path to the anonymizer checkpoint (.pth)",
    )
    parser.add_argument("--pins_dir", type=str, default=PINS_DIR)
    parser.add_argument("--fer_dir", type=str, default=FER_DIR)
    parser.add_argument("--split", type=str, default=SPLIT_PATH,
                        help="Path to split_indices.pt from train_arcface.py")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--arcface_weights", type=str, default='./pretrained/arcface_r50.pth')
    parser.add_argument("--expression_weights", type=str, default='./pretrained/expression_resnet18.pth')
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.arcface_weights:
        cfg.arcface_weights = args.arcface_weights
    if args.expression_weights:
        cfg.expression_weights = args.expression_weights
    if args.device:
        cfg.device = args.device
    out_dir = args.output_dir or cfg.output_dir

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    G = _load_anonymizer(args.checkpoint, cfg, device)

    # ── Test 1: Privacy ──
    arcface = _load_arcface(cfg, device)
    privacy = evaluate_privacy(
        G, arcface,
        pins_dir=args.pins_dir,
        split_path=args.split,
        output_dir=out_dir,
        device=device,
        batch_size=args.batch_size,
    )
    del arcface
    torch.cuda.empty_cache()

    # ── Test 2: Utility ──
    expr_model = _load_expression_model(cfg, device)
    utility = evaluate_utility(
        G, expr_model,
        fer_dir=args.fer_dir,
        output_dir=out_dir,
        device=device,
        batch_size=args.batch_size,
    )

    # ── Summary ──
    po = privacy["orig"]
    pa = privacy["anon"]
    uo = utility["orig"]
    ua = utility["anon"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  PRIVACY (ArcFace re-identification on Pins Face)")
    print(f"    Mean cosine (orig↔anon)  : {privacy['mean_cos']:.4f}")
    print(f"    Original  → Acc {po['accuracy']:5.1f}%  "
          f"P {po['precision']:5.1f}%  R {po['recall']:5.1f}%  F1 {po['f1']:5.1f}%")
    print(f"    Anonymised→ Acc {pa['accuracy']:5.1f}%  "
          f"P {pa['precision']:5.1f}%  R {pa['recall']:5.1f}%  F1 {pa['f1']:5.1f}%")
    print(f"    Drop       → Acc {po['accuracy'] - pa['accuracy']:-5.1f}%  "
          f"F1 {po['f1'] - pa['f1']:-5.1f}%")
    print()
    print("  UTILITY (Expression recognition on FER-2013)")
    print(f"    Original  → Acc {uo['accuracy']:5.1f}%  "
          f"P {uo['precision']:5.1f}%  R {uo['recall']:5.1f}%  F1 {uo['f1']:5.1f}%")
    print(f"    Anonymised→ Acc {ua['accuracy']:5.1f}%  "
          f"P {ua['precision']:5.1f}%  R {ua['recall']:5.1f}%  F1 {ua['f1']:5.1f}%")
    print(f"    Drop       → Acc {uo['accuracy'] - ua['accuracy']:-5.1f}%  "
          f"F1 {uo['f1'] - ua['f1']:-5.1f}%")
    print(f"    Consistency: {utility['consistency']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

