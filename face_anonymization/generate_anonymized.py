"""Generate anonymised images from a directory of originals.

Supports two directory layouts:

1. **Flat** — all images directly inside ``input_dir``:
       input_dir/img001.jpg
       input_dir/img002.jpg

2. **Class-structured** (ImageFolder) — images in per-class subfolders:
       input_dir/angry/img001.jpg
       input_dir/happy/img002.jpg

   The subfolder structure is preserved in the output so that evaluation
   scripts (e.g. ``eval_expression_consistency.py``) can load anonymised
   images with matching labels.

Usage
-----
    # Flat directory (e.g. CelebA-HQ)
    python generate_anonymized.py \
        --checkpoint checkpoints/anonymizer_epoch0010.pth \
        --input_dir  ../celeba_hq_256 \
        --output_dir ../data/anonymised_celeba

    # Class-structured directory (e.g. FER-2013 test set)
    python generate_anonymized.py \
        --checkpoint checkpoints/anonymizer_epoch0010.pth \
        --input_dir  ../expression_recognition/expression_recognition/data/test \
        --output_dir ../expression_recognition/expression_recognition/data/anonymised_test \
        --grayscale
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from models.unet import UNetAnonymizer

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(input_dir: str) -> list[tuple[str, str]]:
    """Return list of (relative_path, absolute_path) for every image.

    Works for both flat and class-structured directories.
    """
    entries: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, input_dir)
                entries.append((rel_path, abs_path))
    return entries


def generate(
    checkpoint: str,
    input_dir: str,
    output_dir: str,
    image_size: int = 256,
    base_channels: int = 64,
    grayscale: bool = False,
    device_str: str = "cuda",
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    G = UNetAnonymizer(base_ch=base_channels).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        G.load_state_dict(ckpt["model"])
    else:
        G.load_state_dict(ckpt)
    G.eval()
    print(f"Loaded anonymizer from {checkpoint}")

    tf_steps: list = []
    if grayscale:
        tf_steps.append(transforms.Grayscale(num_output_channels=3))
    tf_steps += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
    tf = transforms.Compose(tf_steps)

    entries = _collect_images(input_dir)
    print(f"Processing {len(entries)} images → {output_dir}")

    with torch.no_grad():
        for rel_path, abs_path in tqdm(entries, desc="Anonymising"):
            img = Image.open(abs_path).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            anon = G(x).squeeze(0) * 0.5 + 0.5  # back to [0, 1]

            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_image(anon, out_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate anonymised face images",
    )
    parser.add_argument("--checkpoint", default='./checkpoints/anonymizer_epoch0010.pth')
    parser.add_argument("--input_dir", default='./pins_face/105_classes_pins_dataset')
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument(
        "--grayscale", action="store_true",
        help="Convert input images from grayscale to RGB (for FER-2013)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    generate(
        args.checkpoint,
        args.input_dir,
        args.output_dir,
        args.image_size,
        args.base_channels,
        args.grayscale,
        args.device,
    )
