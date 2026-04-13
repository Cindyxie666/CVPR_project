"""Loss functions for face anonymization training.

All four losses are combined into a single ``AnonymizationLoss`` module
which handles internal preprocessing (resize, normalisation) so that the
training loop only needs to pass raw [-1, 1] images.

Losses
------
1. Identity Confusion   – maximise ArcFace embedding distance (adversarial)
2. Expression Preservation – preserve ResNet-18 soft expression predictions
3. Reconstruction        – L1 pixel loss to stabilise training
4. Perceptual            – VGG-19 multi-layer feature matching
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ───────────────────── VGG Feature Extractor ──────────────────────


class VGGFeatureExtractor(nn.Module):
    """Extract intermediate features from a frozen VGG-19 for perceptual loss.

    Returns feature maps after ReLU at layers 2, 7, 12, 21, 30
    (relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 of VGG-19).
    """

    _LAYER_INDICES = [2, 7, 12, 21, 30]

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        slices: list[nn.Sequential] = []
        prev = 0
        for idx in self._LAYER_INDICES:
            slices.append(nn.Sequential(*list(vgg.children())[prev:idx]))
            prev = idx
        self.slices = nn.ModuleList(slices)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats


# ──────────────── Preprocessing helpers (differentiable) ──────────────────


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1]."""
    return x * 0.5 + 0.5


def _imagenet_normalize(x01: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalisation to a [0, 1] tensor."""
    mean = x01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x01 - mean) / std


def prep_arcface(x: torch.Tensor) -> torch.Tensor:
    """Resize to 112×112; ArcFace expects [-1, 1] range."""
    return F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)


def prep_expression(x: torch.Tensor) -> torch.Tensor:
    """Resize to 224×224 with ImageNet normalisation for expression ResNet-18."""
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return _imagenet_normalize(_to_01(x))


def prep_vgg(x: torch.Tensor) -> torch.Tensor:
    """ImageNet normalisation (keep spatial size) for VGG perceptual loss."""
    return _imagenet_normalize(_to_01(x))


# ──────────────── Composite Loss ──────────────────


class AnonymizationLoss(nn.Module):
    """Computes the combined training objective.

    Parameters
    ----------
    arcface : nn.Module
        Frozen face-recognition model that returns 512-d embeddings.
    expression_model : nn.Module
        Frozen expression classifier (logits over expression classes).
    cfg : Config
        Hyperparameters (lambda weights, etc.).
    """

    def __init__(self, arcface: nn.Module, expression_model: nn.Module, cfg):
        super().__init__()
        self.arcface = arcface
        self.expression_model = expression_model
        self.vgg = VGGFeatureExtractor()

        # Freeze all auxiliary models
        for model in (self.arcface, self.expression_model, self.vgg):
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        self.cfg = cfg

    def train(self, mode: bool = True) -> "AnonymizationLoss":
        """Override to prevent frozen models from being switched to train mode.

        Without this, calling `criterion.train()` would recursively enable
        Dropout in the expression model and switch BatchNorm to training
        statistics in ArcFace — silently corrupting both.
        """
        self.training = mode
        return self

    # ------------------------------------------------------------------

    def forward(
        self, original: torch.Tensor, anonymized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        original, anonymized : Tensor  (B, 3, 256, 256) in [-1, 1]

        Returns
        -------
        dict with keys: identity, expression, reconstruction, perceptual, total
        """
        losses: dict[str, torch.Tensor] = {}

        # 1. Identity confusion (adversarial): minimise cosine similarity
        with torch.no_grad():
            emb_orig = self.arcface(prep_arcface(original))
        emb_anon = self.arcface(prep_arcface(anonymized))
        losses["identity"] = F.cosine_similarity(emb_orig, emb_anon, dim=1).mean()

        # 2. Expression preservation: KL divergence on soft predictions
        with torch.no_grad():
            expr_target = F.softmax(
                self.expression_model(prep_expression(original)), dim=1,
            )
        expr_anon_logits = self.expression_model(prep_expression(anonymized))
        losses["expression"] = F.kl_div(
            F.log_softmax(expr_anon_logits, dim=1),
            expr_target,
            reduction="batchmean",
        )

        # 3. Reconstruction (L1)
        losses["reconstruction"] = F.l1_loss(anonymized, original)

        # 4. Perceptual (VGG-19 multi-layer MSE)
        with torch.no_grad():
            vgg_orig = self.vgg(prep_vgg(original))
        vgg_anon = self.vgg(prep_vgg(anonymized))
        losses["perceptual"] = sum(
            F.mse_loss(fo, fa) for fo, fa in zip(vgg_orig, vgg_anon)
        )

        # Weighted total
        c = self.cfg
        losses["total"] = (
            c.lambda_identity * losses["identity"]
            + c.lambda_expression * losses["expression"]
            + c.lambda_reconstruction * losses["reconstruction"]
            + c.lambda_perceptual * losses["perceptual"]
        )
        return losses
