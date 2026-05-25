"""
Vision Transformer models for image classification.

FineTuneViT  — ImageNet-pretrained torchvision ViT backbones (vit_b_16, …).
TinyViT      — compact ViT trained from scratch on small images (32×32 CIFAR).

Fine-tuning strategies (FineTuneViT)
------------------------------------
full        — all parameters updated
head_only   — only the classification head is updated (linear probe)
last_n      — last N encoder blocks + head are updated (partial fine-tune)

Usage example
-------------
    model = FineTuneViT(
        variant="vit_b_16",
        num_classes=10,
        pretrained=True,
        strategy="last_n",
        trainable_blocks=4,
        dropout=0.0,
    )
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from torchvision import models
    from torchvision.models import (
        ViT_B_16_Weights,
        ViT_B_32_Weights,
        ViT_L_16_Weights,
    )
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


_BACKBONE_MAP = {
    "vit_b_16": ("vit_b_16", 768),
    "vit_b_32": ("vit_b_32", 768),
    "vit_l_16": ("vit_l_16", 1024),
}

_WEIGHTS_MAP = {
    "vit_b_16": ViT_B_16_Weights.DEFAULT if _TORCHVISION_AVAILABLE else None,
    "vit_b_32": ViT_B_32_Weights.DEFAULT if _TORCHVISION_AVAILABLE else None,
    "vit_l_16": ViT_L_16_Weights.DEFAULT if _TORCHVISION_AVAILABLE else None,
}


class FineTuneViT(nn.Module):
    """
    Pretrained torchvision ViT with a replaceable classification head.

    Parameters
    ----------
    variant : str
        One of 'vit_b_16', 'vit_b_32', 'vit_l_16'.
    num_classes : int
        Number of output classes.
    pretrained : bool
        If True, load ImageNet-pretrained weights.
    strategy : str
        Fine-tuning strategy: 'full', 'head_only', or 'last_n'.
    trainable_blocks : int
        Number of transformer encoder blocks to unfreeze when
        strategy='last_n'.
    dropout : float
        Dropout rate applied before the linear head (0 = disabled).
    """

    def __init__(
        self,
        variant: str = "vit_b_16",
        num_classes: int = 10,
        pretrained: bool = True,
        strategy: str = "full",
        trainable_blocks: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not _TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required. Install with: pip install torchvision"
            )
        if variant not in _BACKBONE_MAP:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from {list(_BACKBONE_MAP)}"
            )
        if strategy not in ("full", "head_only", "last_n"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose from 'full', 'head_only', 'last_n'."
            )

        builder_name, hidden_dim = _BACKBONE_MAP[variant]
        weights = _WEIGHTS_MAP[variant] if pretrained else None
        # Pretrained weights require the default 1000-class head; replace it below.
        if pretrained:
            backbone = getattr(models, builder_name)(weights=weights)
        else:
            backbone = getattr(models, builder_name)(
                weights=None, num_classes=num_classes
            )

        backbone.heads = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.backbone = backbone
        self.variant  = variant
        self.strategy = strategy

        self._apply_strategy(strategy, trainable_blocks)

    def _encoder_blocks(self) -> list[nn.Module]:
        return list(self.backbone.encoder.layers.children())

    def _apply_strategy(self, strategy: str, trainable_blocks: int) -> None:
        if strategy == "full":
            return

        for p in self.backbone.parameters():
            p.requires_grad = False

        if strategy == "head_only":
            for p in self.backbone.heads.parameters():
                p.requires_grad = True

        elif strategy == "last_n":
            blocks = self._encoder_blocks()
            n = max(1, min(trainable_blocks, len(blocks)))
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.backbone.encoder.ln.parameters():
                p.requires_grad = True
            for p in self.backbone.heads.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_num_params(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    def parameter_groups(
        self,
        head_lr_scale: float = 10.0,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        head_params    = list(self.backbone.heads.parameters())
        head_param_ids = {id(p) for p in head_params}
        backbone_params = [
            p for p in self.parameters()
            if id(p) not in head_param_ids and p.requires_grad
        ]
        groups = []
        if backbone_params:
            groups.append({
                "params":       backbone_params,
                "lr_scale":     1.0,
                "weight_decay": weight_decay,
            })
        if head_params:
            groups.append({
                "params":       head_params,
                "lr_scale":     head_lr_scale,
                "weight_decay": weight_decay,
            })
        return groups


# ---------------------------------------------------------------------------
# TinyViT — from-scratch ViT for 28×28 / 32×32 inputs
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """
    Compact Vision Transformer for small images (CIFAR / MNIST).

    Variants
    --------
    tiny   — embed_dim=192, depth=12, heads=3
    small  — embed_dim=384, depth=12, heads=6
    base   — embed_dim=512, depth=12, heads=8
    """

    _VARIANTS = {
        "tiny":  (192, 12, 3),
        "small": (384, 12, 6),
        "base":  (512, 12, 8),
    }

    def __init__(
        self,
        img_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 10,
        variant: str = "small",
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from {list(self._VARIANTS)}"
            )
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )

        embed_dim, depth, num_heads = self._VARIANTS[variant]
        self.patch_embed = _PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                _TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                nn.init.trunc_normal_(m.weight, std=math.sqrt(1.0 / fan_in))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
