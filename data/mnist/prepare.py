"""
Prepare the MNIST dataset.

Downloads the dataset via torchvision (cached in the same directory),
computes per-channel mean/std on the training split, and writes a
`meta.json` file that `train_image.py` reads at startup.

Usage:
    python data/mnist/prepare.py
    python data/mnist/prepare.py --data_dir /path/to/cache --val_fraction 0.1
"""
import os
import json
import argparse

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Prepare MNIST dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory where the raw dataset will be cached (default: data/mnist/)",
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
    print("[*] Downloading MNIST (if not already cached)...")
    raw_transform = T.ToTensor()
    train_full = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, download=True, transform=raw_transform
    )
    test_set = torchvision.datasets.MNIST(
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
    # Compute normalisation stats from the training split only
    # ------------------------------------------------------------------
    print("[*] Computing normalisation statistics...")
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=1024, num_workers=0, shuffle=False
    )
    pixel_sum   = torch.zeros(1)
    pixel_sq    = torch.zeros(1)
    n_pixels    = 0
    for imgs, _ in loader:
        # imgs: (B, 1, 28, 28)
        pixel_sum += imgs.sum()
        pixel_sq  += (imgs ** 2).sum()
        n_pixels  += imgs.numel()

    mean = (pixel_sum / n_pixels).item()
    std  = ((pixel_sq / n_pixels - mean ** 2) ** 0.5).item()
    print(f"    mean = {mean:.6f}")
    print(f"    std  = {std:.6f}")

    # ------------------------------------------------------------------
    # Class info
    # ------------------------------------------------------------------
    classes = [str(i) for i in range(10)]

    # ------------------------------------------------------------------
    # Write meta.json
    # ------------------------------------------------------------------
    meta = {
        "dataset":       "mnist",
        "num_classes":   10,
        "classes":       classes,
        "image_size":    [1, 28, 28],   # C x H x W
        "n_train":       n_train,
        "n_val":         n_val,
        "n_test":        len(test_set),
        "val_fraction":  args.val_fraction,
        "mean":          [mean],        # single-channel
        "std":           [std],
    }
    meta_path = os.path.join(args.data_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] Metadata written to {meta_path}")
    print("[✓] MNIST preparation complete.")


if __name__ == "__main__":
    main()
