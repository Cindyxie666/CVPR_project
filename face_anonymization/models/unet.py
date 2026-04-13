"""U-Net anonymization network.

Encoder–bottleneck–decoder with skip connections.
Input and output are 256×256×3 images in [-1, 1] (tanh output).
Skip connections preserve spatial/expression structure while the
bottleneck disrupts identity-specific features.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class EncoderBlock(nn.Module):
    """Strided convolution with optional normalization."""

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Transposed convolution → concat skip → conv refine."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.norm_up = nn.InstanceNorm2d(out_ch)
        self.act_up = nn.ReLU(inplace=True)

        self.refine = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act_up(self.norm_up(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


class UNetAnonymizer(nn.Module):
    """
    U-Net for face anonymization.

    Architecture (default base_ch=64, input 256×256):

        Encoder                        Decoder
        ─────────────────────────      ─────────────────────────
        e1  3→64   256→128   ─skip1──▶ d3  128→64+64   →64   128→256
        e2  64→128  128→64   ─skip2──▶ d2  256→128+128 →128  64→128
        e3  128→256 64→32    ─skip3──▶ d1  512→256+256 →256  32→64 (sic: 16→32)
        e4  256→512 32→16
                 ↓
            bottleneck (3× ResBlock, 512ch, 16×16)
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        c = base_ch

        # --- Encoder ---
        self.enc1 = EncoderBlock(in_channels, c, use_norm=False)  # 256→128
        self.enc2 = EncoderBlock(c, c * 2)                        # 128→64
        self.enc3 = EncoderBlock(c * 2, c * 4)                    # 64→32
        self.enc4 = EncoderBlock(c * 4, c * 8)                    # 32→16

        # --- Bottleneck (identity disruption happens here) ---
        self.bottleneck = nn.Sequential(
            ResidualBlock(c * 8),
            ResidualBlock(c * 8),
            ResidualBlock(c * 8),
        )

        # --- Decoder with skip connections ---
        self.dec1 = DecoderBlock(c * 8, c * 4, skip_ch=c * 4)   # 16→32,  skip=enc3
        self.dec2 = DecoderBlock(c * 4, c * 2, skip_ch=c * 2)   # 32→64,  skip=enc2
        self.dec3 = DecoderBlock(c * 2, c,     skip_ch=c)        # 64→128, skip=enc1

        # --- Final upsample (no skip) ---
        self.final = nn.Sequential(
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, in_channels, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)   # 128×128
        e2 = self.enc2(e1)  # 64×64
        e3 = self.enc3(e2)  # 32×32
        e4 = self.enc4(e3)  # 16×16

        b = self.bottleneck(e4)

        d1 = self.dec1(b, e3)    # 32×32
        d2 = self.dec2(d1, e2)   # 64×64
        d3 = self.dec3(d2, e1)   # 128×128

        return self.final(d3)    # 256×256, range [-1, 1]
