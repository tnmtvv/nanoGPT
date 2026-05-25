"""
Prepare the CIFAR-10 dataset.

Downloads the dataset via torchvision (cached in the same directory),
computes per-channel (RGB) mean/std on the training split, and writes a
`meta.json` file that `train_image.py` reads at startup.

Usage:
    python data/cifar10/prepare.py
    python data/cifar10/prepare.py --data_dir /path/to/cache --val_fraction 0.1
"""
import os
import json
import argparse

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory where the raw dataset will be cached (default: data/cifar10/)",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of training data to use as validation (default: 0.1)",
    )
    args = parser.parse_args()

    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        raise ImportError(
            "torchvision is required. Install with: pip install torchvision"
        )

    os.makedirs(args.data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    print("[*] Downloading CIFAR-10 (if not already cached)...")
    raw_transform = T.ToTensor()
    train_full = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=raw_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=raw_transform
    )
    print(f"    Train+val examples : {len(train_full):,}")
    print(f"    Test  examples     : {len(test_set):,}")

    # ------------------------------------------------------------------
    # Train / val split (deterministic)
    # ------------------------------------------------------------------
    n_total = len(train_full)
    n_val   = int(n_total * args.val_fraction)
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        train_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"    Train split        : {n_train:,}")
    print(f"    Val   split        : {n_val:,}")

    # ------------------------------------------------------------------
    # Compute per-channel normalisation stats from training split only
    # ------------------------------------------------------------------
    print("[*] Computing per-channel normalisation statistics...")
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=512, num_workers=0, shuffle=False
    )
    # Accumulate over C=3 channels
    pixel_sum = torch.zeros(3)
    pixel_sq  = torch.zeros(3)
    n_pixels  = 0
    for imgs, _ in loader:
        # imgs: (B, 3, 32, 32)
        b = imgs.size(0)
        imgs_flat = imgs.view(b, 3, -1)          # (B, 3, H*W)
        pixel_sum += imgs_flat.sum(dim=[0, 2])
        pixel_sq  += (imgs_flat ** 2).sum(dim=[0, 2])
        n_pixels  += b * imgs.size(2) * imgs.size(3)

    mean = (pixel_sum / n_pixels).tolist()
    std  = ((pixel_sq / n_pixels - torch.tensor(mean) ** 2) ** 0.5).tolist()
    print(f"    mean (R,G,B) = {[f'{m:.6f}' for m in mean]}")
    print(f"    std  (R,G,B) = {[f'{s:.6f}' for s in std]}")

    # ------------------------------------------------------------------
    # Class info
    # ------------------------------------------------------------------
    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    # ------------------------------------------------------------------
    # Write meta.json
    # ------------------------------------------------------------------
    meta = {
        "dataset":      "cifar10",
        "num_classes":  10,
        "classes":      classes,
        "image_size":   [3, 32, 32],
        "n_train":      n_train,
        "n_val":        n_val,
        "n_test":       len(test_set),
        "val_fraction": args.val_fraction,
        "mean":         mean,
        "std":          std,
    }
    meta_path = os.path.join(args.data_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] Metadata written to {meta_path}")
    print("[✓] CIFAR-10 preparation complete.")


if __name__ == "__main__":
    main()
