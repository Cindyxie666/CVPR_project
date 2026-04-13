from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # ---------- Data ----------
    data_dir: str = "./celeba_hq_256"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4

    # ---------- U-Net ----------
    base_channels: int = 64

    # ---------- Pretrained model paths ----------
    # Point these to the teammate's saved weights when available.
    # ArcFace: the IResNet-50 backbone (512-d face embeddings).
    arcface_weights: str = "./pretrained/arcface_r50.pth"
    # Expression ResNet-18 trained on FER-2013 (7-class).
    expression_weights: str = "./pretrained/expression_resnet18.pth"
    expression_num_classes: int = 7

    # ---------- Loss weights ----------
    lambda_identity: float = 3.0
    lambda_expression: float = 2.0
    lambda_reconstruction: float = 0.05
    lambda_perceptual: float = 0.5

    # ---------- Training ----------
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    epochs: int = 20
    device: str = "cuda"
    resume: str = "./checkpoints/anonymizer_epoch0010.pth"

    # ---------- Output ----------
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1
    sample_every: int = 1
