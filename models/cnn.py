"""
Custom CNN architectures for image classification.

LeNet5   — classic 5-layer network; works well on MNIST (28×28 greyscale).
ConvNet  — modern BN-equipped ConvNet; works well on CIFAR-10/100 (32×32 RGB).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LeNet-5  (adapted for variable input channels and number of classes)
# ---------------------------------------------------------------------------

class LeNet5(nn.Module):
    """
    LeCun et al. 1998-style architecture adapted for modern PyTorch.
    Original target: MNIST 1×28×28 → 10 classes.
    Can also be used for 3-channel inputs by passing in_channels=3.

    Architecture:
        Conv(6, 5×5) → AvgPool(2) → Conv(16, 5×5) → AvgPool(2)
        → Flatten → FC(120) → FC(84) → FC(num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),  # 28→28, 32→32
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),                # 28→14, 32→16
            nn.Conv2d(6, 16, kernel_size=5),                      # 14→10, 16→12
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),                # 10→5,  12→6
        )
        # Compute the flattened feature size dynamically
        self._flat_features = self._get_flat_features(in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_features, 120),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(84, num_classes),
        )
        self._init_weights()

    def _get_flat_features(self, in_channels: int) -> int:
        """Run a dummy forward pass to find the flattened feature dimension."""
        # Assume MNIST-like 28×28 as baseline; ConvNet handles 32×32.
        dummy = torch.zeros(1, in_channels, 28, 28)
        out = self.features(dummy)
        return int(out.numel())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="tanh")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# ConvNet  — designed for 3×32×32 CIFAR inputs
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv → BN → ReLU (→ optional MaxPool)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        pool: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvNet(nn.Module):
    """
    A 9-layer ConvNet with BatchNorm and residual-style skip connections,
    tuned for 3×32×32 CIFAR inputs.

    Architecture (default):
        [Conv64] × 2 → Pool(2) → [Conv128] × 2 → Pool(2)
        → [Conv256] × 2 → Pool(2) → GAP → FC(num_classes)

    The depth is controlled by the `variant` argument:
        'small'   — 2+2+2 conv layers, [32,64,128] channels  (good for CIFAR-10)
        'medium'  — 2+2+2 conv layers, [64,128,256] channels (good for CIFAR-10/100)
        'large'   — 3+3+3 conv layers, [64,128,256] channels (good for CIFAR-100)
    """

    _VARIANTS = {
        #  name     : (channels,       blocks_per_stage)
        "small":   ([32, 64, 128],    [2, 2, 2]),
        "medium":  ([64, 128, 256],   [2, 2, 2]),
        "large":   ([64, 128, 256],   [3, 3, 3]),
    }

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        variant: str = "medium",
        dropout: float = 0.0,
    ):
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {list(self._VARIANTS)}"
            )
        channels, blocks_per_stage = self._VARIANTS[variant]

        stages: list[nn.Module] = []
        c_in = in_channels
        for stage_idx, (c_out, n_blocks) in enumerate(
            zip(channels, blocks_per_stage)
        ):
            for blk_idx in range(n_blocks):
                pool = (blk_idx == n_blocks - 1)          # pool on last block of each stage
                do   = dropout if blk_idx == n_blocks - 1 else 0.0
                stages.append(_ConvBlock(c_in, c_out, pool=pool, dropout=do))
                c_in = c_out
        self.features   = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head       = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.head(x)
        return x

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
