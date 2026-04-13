"""ArcFace classification head (Additive Angular Margin Loss).

Implements the margin-based softmax from:
  Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
  Recognition," CVPR 2019.

During training this replaces a standard Linear+CrossEntropy classifier.
At inference time only the backbone is used to extract embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive Angular Margin softmax head.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the input embeddings (typically 512).
    num_classes : int
        Number of identity classes.
    s : float
        Feature re-scale factor (logit scale).  Default 64.0.
    m : float
        Angular margin in radians.  Default 0.5 (~28.6°).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 105,
        s: float = 64.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute margin terms
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Threshold to fall back to the monotonic region
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : (B, D)  L2-normalised face embeddings.
        labels     : (B,)    ground-truth identity indices.

        Returns
        -------
        logits : (B, C)  scaled cosine logits with angular margin applied
                         to the target class.
        """
        # Normalise weight vectors
        w = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity → (B, C)
        cosine = F.linear(embeddings, w)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt(1.0 - cosine * cosine)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical safety: when cos(theta) < cos(pi - m), use a linear
        # fallback so that the function stays monotonically decreasing.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only to the ground-truth class
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine

        return logits * self.s
