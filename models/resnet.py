"""
Pretrained ResNet fine-tuning wrappers.

Supported backbones: resnet18, resnet34, resnet50, resnet101, resnet152.

Fine-tuning strategies
----------------------
full        — all parameters updated (best for small datasets with large LR)
head_only   — only the classifier head is updated (linear probe)
last_n      — last N residual stages + head are updated (partial fine-tune)

Usage example
-------------
    model = FineTuneResNet(
        variant="resnet18",
        num_classes=10,
        pretrained=True,
        strategy="full",
        dropout=0.0,
    )
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision import models
    from torchvision.models import (
        ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
        ResNet101_Weights, ResNet152_Weights,
    )
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


_BACKBONE_MAP = {
    "resnet18":  ("resnet18",  512),
    "resnet34":  ("resnet34",  512),
    "resnet50":  ("resnet50",  2048),
    "resnet101": ("resnet101", 2048),
    "resnet152": ("resnet152", 2048),
}

_WEIGHTS_MAP = {
    "resnet18":  ResNet18_Weights.DEFAULT  if _TORCHVISION_AVAILABLE else None,
    "resnet34":  ResNet34_Weights.DEFAULT  if _TORCHVISION_AVAILABLE else None,
    "resnet50":  ResNet50_Weights.DEFAULT  if _TORCHVISION_AVAILABLE else None,
    "resnet101": ResNet101_Weights.DEFAULT if _TORCHVISION_AVAILABLE else None,
    "resnet152": ResNet152_Weights.DEFAULT if _TORCHVISION_AVAILABLE else None,
}


class FineTuneResNet(nn.Module):
    """
    Pretrained ResNet with a replaceable classification head.

    Parameters
    ----------
    variant : str
        One of 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
    num_classes : int
        Number of output classes.
    pretrained : bool
        If True, load ImageNet-pretrained weights.
    strategy : str
        Fine-tuning strategy: 'full', 'head_only', or 'last_n'
        (where n is set via `trainable_stages`).
    trainable_stages : int
        Number of ResNet stages (layer1–layer4) to unfreeze when
        strategy='last_n'. Valid range: 1–4.
    dropout : float
        Dropout rate applied before the linear head (0 = disabled).
    small_input : bool
        If True, replaces the first 7×7 conv (stride 2) and MaxPool with a
        3×3 conv (stride 1) — recommended for 32×32 CIFAR inputs so spatial
        resolution is not immediately halved down to 4×4.
    """

    def __init__(
        self,
        variant: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = True,
        strategy: str = "full",
        trainable_stages: int = 2,
        dropout: float = 0.0,
        small_input: bool = False,
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

        builder_name, feat_dim = _BACKBONE_MAP[variant]
        weights = _WEIGHTS_MAP[variant] if pretrained else None
        backbone = getattr(models, builder_name)(weights=weights)

        # ── Adapt first layer for small (32×32) inputs ──────────────────────
        if small_input:
            backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            backbone.maxpool = nn.Identity()   # remove stride-2 pooling

        # ── Replace classification head ──────────────────────────────────────
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

        self.backbone = backbone
        self.variant  = variant
        self.strategy = strategy

        # ── Freeze / unfreeze parameters ─────────────────────────────────────
        self._apply_strategy(strategy, trainable_stages)

    def _apply_strategy(self, strategy: str, trainable_stages: int) -> None:
        """Freeze parameters according to the chosen fine-tuning strategy."""
        if strategy == "full":
            # All parameters trainable — nothing to do.
            return

        # Freeze everything first.
        for p in self.backbone.parameters():
            p.requires_grad = False

        if strategy == "head_only":
            # Only unfreeze the new FC head.
            for p in self.backbone.fc.parameters():
                p.requires_grad = True

        elif strategy == "last_n":
            # Unfreeze the last `trainable_stages` residual stages + head.
            stages = [
                self.backbone.layer4,
                self.backbone.layer3,
                self.backbone.layer2,
                self.backbone.layer1,
            ]
            n = max(1, min(trainable_stages, 4))
            for stage in stages[:n]:
                for p in stage.parameters():
                    p.requires_grad = True
            for p in self.backbone.fc.parameters():
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
        """
        Return parameter groups with a higher LR for the head and a lower LR
        for the backbone. Multiply the base LR by `head_lr_scale` for the head.

        Usage::

            groups = model.parameter_groups(head_lr_scale=10.0)
            optimizer = torch.optim.AdamW(
                [{"params": g["params"], "lr": base_lr * g["lr_scale"],
                  "weight_decay": g["weight_decay"]}
                 for g in groups]
            )
        """
        head_params    = list(self.backbone.fc.parameters())
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
