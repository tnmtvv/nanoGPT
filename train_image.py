"""
Image Classification Training Script — MNIST / CIFAR-10 / CIFAR-100.

Supports:
  - Training custom CNNs (LeNet5, ConvNet) from scratch
  - Fine-tuning pretrained ResNet models (resnet18/34/50/101/152)
  - ClearML experiment tracking
  - Local CSV logging (same CsvLogger used in train.py)
  - Cosine LR schedule with linear warmup
  - Mixed-precision training (torch.amp)
  - Checkpoint save/resume

Quick start
-----------
  # Prepare dataset first:
  python data/mnist/prepare.py
  python data/cifar10/prepare.py
  python data/cifar100/prepare.py

  # Train LeNet-5 on MNIST:
  python train_image.py --config config/mnist_cnn.py

  # Fine-tune ResNet-18 on CIFAR-10:
  python train_image.py --config config/cifar10_resnet18.py

  # Override a config value on the command line:
  python train_image.py --config config/cifar10_resnet18.py --learning_rate=0.01

  # Enable ClearML logging:
  python train_image.py --config config/cifar10_resnet18.py --clearml_log=True
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adagram_optimizers.AdagramPS import AdaGramPS
from adagram_optimizers.AdamGram import AdamGram, SymAdamGram, EQAdamGram, SVDAdamGram
from muon import MuonWithAuxAdam


# ---------------------------------------------------------------------------
# Default hyper-parameters (all can be overridden via config file or CLI)
# ---------------------------------------------------------------------------

# I/O
out_dir                     = "out-image"
eval_interval               = 1
log_interval                = 1
always_save_checkpoint      = False
init_from                   = "scratch"      # 'scratch' | 'resume' | 'pretrained'

# ClearML / logging
clearml_log                 = False
clearml_project             = "image-classification"
clearml_task_name           = "run"
wandb_log                   = False

# Dataset
dataset                     = "cifar10"      # 'mnist' | 'cifar10' | 'cifar100'
val_fraction                = 0.1
num_workers                 = 4
augment                     = True

# Model
model_type                  = "resnet"       # 'cnn' | 'resnet' | 'vit'
model_variant               = "resnet18"
finetune_strategy           = "full"         # 'full' | 'head_only' | 'last_n'
trainable_stages            = 2
trainable_blocks            = 4
small_input                 = True
dropout                     = 0.0
resize_to                   = None

# Training
batch_size                  = 128
max_epochs                  = 30
gradient_accumulation_steps = 1
weight_decay                = 1e-4
learning_rate               = 1e-3
beta1                       = 0.9
beta2                       = 0.999
grad_clip                   = 1.0
seed                        = 42

# Optimiser
optimizer_type              = "adamw"        # 'adamw' | 'adam' | 'sgd' | 'rmsprop' | 'muonwithadamw' | 'adagram' | 'adamgram'
momentum                    = 0.9
nesterov                    = True
rank                        = 1

# LR schedule
decay_lr                    = True
warmup_epochs               = 3
min_lr                      = 1e-5

# System
device                      = "cuda" if torch.cuda.is_available() else "cpu"
dtype                       = "bfloat16"    # 'float32' | 'float16' | 'bfloat16'
compile_model               = False


# ---------------------------------------------------------------------------
# Helpers: distributed
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_distributed() -> None:
    """
    Bootstrap a trivial single-process group when running without torchrun.
    Required by MuonWithAuxAdam, which internally calls dist.get_world_size().
    When launched via `torchrun --nproc_per_node=N`, the env vars RANK /
    WORLD_SIZE / MASTER_ADDR are already set and init_process_group() is a
    no-op here (guarded by is_initialized()).
    """
    import torch.distributed as dist
    if dist.is_available() and not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # torchrun path — let PyTorch read env vars automatically
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        else:
            # Single-process path (plain `python train_image.py`)
            port = _find_free_port()
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=1,
                rank=0,
            )
        print(f"[*] Distributed process group initialised "
              f"(world_size={dist.get_world_size()}, rank={dist.get_rank()})")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image classification training (MNIST / CIFAR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",             type=str,   default=None)
    parser.add_argument("--out_dir",            type=str)
    parser.add_argument("--dataset",            type=str)
    parser.add_argument("--model_type",         type=str)
    parser.add_argument("--model_variant",      type=str)
    parser.add_argument("--finetune_strategy",  type=str)
    parser.add_argument("--trainable_stages",   type=int)
    parser.add_argument("--small_input",        type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--dropout",            type=float)
    parser.add_argument("--batch_size",         type=int)
    parser.add_argument("--max_epochs",         type=int)
    parser.add_argument("--learning_rate",      type=float)
    parser.add_argument("--weight_decay",       type=float)
    parser.add_argument("--grad_clip",          type=float)
    parser.add_argument("--seed",               type=int)
    parser.add_argument("--device",             type=str)
    parser.add_argument("--dtype",              type=str,   choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--compile_model",      type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--init_from",          type=str,   choices=["scratch", "resume", "pretrained"])
    parser.add_argument("--trainable_blocks",   type=int)
    parser.add_argument("--resize_to",          type=int)
    parser.add_argument("--optimizer_type",     type=str)
    parser.add_argument("--momentum",           type=float)
    parser.add_argument("--nesterov",           type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--clearml_log",        type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--clearml_project",    type=str)
    parser.add_argument("--clearml_task_name",  type=str)
    parser.add_argument("--augment",            type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--eval_interval",      type=int)
    parser.add_argument("--always_save_checkpoint", type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--rank", type=int, default=None)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    g = globals()
    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is not None:
            g[key] = val
    if g.get("wandb_log"):
        g["clearml_log"] = True


def apply_config(config_path: str) -> None:
    print(f"[*] Loading config: {config_path}")
    with open(config_path) as f:
        content = f.read()
    print(content)
    exec(content, globals())   # noqa: S102


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _get_transforms(dataset_name: str, meta: dict, augment: bool,
                    resize_to: Optional[int] = None):
    try:
        import torchvision.transforms as T
    except ImportError:
        raise ImportError("torchvision is required: pip install torchvision")

    mean = meta["mean"]
    std  = meta["std"]
    in_channels = meta["image_size"][0]

    if resize_to is not None:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    normalize = T.Normalize(mean=mean, std=std)
    to_rgb = (T.Lambda(lambda x: x.repeat(3, 1, 1))
              if (in_channels == 1 and resize_to is not None)
              else nn.Identity())

    if resize_to is not None:
        train_resize = T.Compose([T.Resize(int(resize_to * 1.14)), T.RandomCrop(resize_to)])
        val_resize   = T.Resize(resize_to)
    else:
        train_resize = None
        val_resize   = None

    if dataset_name == "mnist" and resize_to is None:
        val_tf = T.Compose([T.ToTensor(), normalize])
        train_tf = T.Compose([T.RandomCrop(28, padding=4), T.ToTensor(), normalize]) if augment else val_tf
    elif dataset_name == "mnist" and resize_to is not None:
        val_tf   = T.Compose([val_resize, T.ToTensor(), to_rgb, normalize])
        train_tf = (T.Compose([train_resize, T.RandomHorizontalFlip(), T.ToTensor(), to_rgb, normalize])
                    if augment else val_tf)
    else:
        if resize_to is None:
            val_tf = T.Compose([T.ToTensor(), normalize])
            train_tf = (T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(), normalize,
            ]) if augment else val_tf)
        else:
            val_tf = T.Compose([val_resize, T.ToTensor(), normalize])
            train_tf = (T.Compose([
                train_resize, T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(), normalize,
            ]) if augment else val_tf)

    return train_tf, val_tf


def build_dataloaders(
    dataset_name: str,
    meta: dict,
    batch_size: int,
    val_fraction: float,
    augment: bool,
    num_workers: int,
    seed: int,
    resize_to: Optional[int] = None,
):
    try:
        import torchvision.datasets as dsets
    except ImportError:
        raise ImportError("torchvision is required: pip install torchvision")

    data_dir         = os.path.join("data", dataset_name)
    train_tf, val_tf = _get_transforms(dataset_name, meta, augment, resize_to=resize_to)

    ds_map = {
        "mnist":    (dsets.MNIST,    dsets.MNIST),
        "cifar10":  (dsets.CIFAR10,  dsets.CIFAR10),
        "cifar100": (dsets.CIFAR100, dsets.CIFAR100),
    }
    TrainCls, TestCls = ds_map[dataset_name]

    train_full = TrainCls(root=data_dir, train=True,  download=False, transform=train_tf)
    val_full   = TrainCls(root=data_dir, train=True,  download=False, transform=val_tf)
    test_ds    = TestCls(root=data_dir,  train=False, download=False, transform=val_tf)

    n_total = len(train_full)
    n_val   = int(n_total * val_fraction)
    n_train = n_total - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_idx, val_idx = torch.utils.data.random_split(range(n_total), [n_train, n_val], generator=gen)
    train_idx = list(train_idx.indices)
    val_idx   = list(val_idx.indices)

    train_ds = torch.utils.data.Subset(train_full, train_idx)
    val_ds   = torch.utils.data.Subset(val_full,   val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    model_type: str,
    model_variant: str,
    num_classes: int,
    in_channels: int,
    init_from: str,
    finetune_strategy: str,
    trainable_stages: int,
    trainable_blocks: int,
    small_input: bool,
    dropout: float,
) -> nn.Module:
    if model_type == "cnn":
        from models.cnn import LeNet5, ConvNet
        if model_variant == "lenet5":
            model = LeNet5(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
        else:
            model = ConvNet(in_channels=in_channels, num_classes=num_classes,
                            variant=model_variant, dropout=dropout)
        print(f"[*] CNN ({model_variant}) — {model.get_num_params():,} trainable params")

    elif model_type == "resnet":
        from models.resnet import FineTuneResNet
        model = FineTuneResNet(
            variant=model_variant,
            num_classes=num_classes,
            pretrained=(init_from == "pretrained"),
            strategy=finetune_strategy,
            trainable_stages=trainable_stages,
            dropout=dropout,
            small_input=small_input,
        )
        total     = model.get_num_params(trainable_only=False)
        trainable = model.get_num_params(trainable_only=True)
        print(f"[*] ResNet ({model_variant}, strategy={finetune_strategy}) — "
              f"{trainable:,} / {total:,} params trainable")

    elif model_type == "vit":
        from models.vit import FineTuneViT, TinyViT
        _pretrained_variants = ("vit_b_16", "vit_b_32", "vit_l_16")
        if model_variant in _pretrained_variants:
            model = FineTuneViT(
                variant=model_variant,
                num_classes=num_classes,
                pretrained=(init_from == "pretrained"),
                strategy=finetune_strategy,
                trainable_blocks=trainable_blocks,
                dropout=dropout,
            )
            total     = model.get_num_params(trainable_only=False)
            trainable = model.get_num_params(trainable_only=True)
            print(f"[*] ViT ({model_variant}, strategy={finetune_strategy}) — "
                  f"{trainable:,} / {total:,} params trainable")
        else:
            model = TinyViT(
                img_size=32 if in_channels == 3 else 28,
                in_channels=in_channels,
                num_classes=num_classes,
                variant=model_variant,
                dropout=dropout,
            )
            print(f"[*] TinyViT ({model_variant}) — {model.get_num_params():,} trainable params")

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'cnn', 'resnet', or 'vit'.")

    return model


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    momentum: float,
    nesterov: bool,
    model_type: str,
    rank=None
) -> torch.optim.Optimizer:
    if model_type in ("resnet", "vit") and hasattr(model, "parameter_groups"):
        groups_meta  = model.parameter_groups(head_lr_scale=10.0, weight_decay=weight_decay)
        param_groups = [
            {"params": g["params"], "lr": learning_rate * g["lr_scale"], "weight_decay": g["weight_decay"]}
            for g in groups_meta
        ]
        print(f"[*] Differential LR — backbone: {learning_rate:.2e}, head: {learning_rate * 10:.2e}")
    else:
        decay   = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        param_groups = [
            {"params": decay,   "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]

    opt = optimizer_type.lower()
    if opt == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(beta1, beta2))
    elif opt == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=learning_rate, betas=(beta1, beta2))
    elif opt == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=learning_rate,
                                    momentum=momentum, nesterov=nesterov)
    elif opt == "rmsprop":
        optimizer = torch.optim.RMSprop(param_groups, lr=learning_rate, momentum=momentum)
    elif opt == "muonwithadamw":
        # dist process group is already initialized in main() — safe to use here
        decay_params   = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        muon_groups = [
            dict(params=decay_params,   use_muon=True,  lr=learning_rate, weight_decay=weight_decay),
            dict(params=nodecay_params, use_muon=False, lr=learning_rate,
                 betas=(beta1, beta2), weight_decay=weight_decay),
        ]
        optimizer = MuonWithAuxAdam(muon_groups)
    elif opt == "adagram":
        print()
        print()
        print()
        print(f"rank = {rank} !!!!")
        print()
        print()
        print()
        optimizer = AdaGramPS(param_groups, lr=learning_rate, max_rank=rank)
    elif opt == "adamgram":
        print()
        print()
        print()
        print(f"rank = {rank} !!!!!")
        print()
        print()
        print()
        optimizer = AdamGram(param_groups, lr=learning_rate, max_rank=rank)
    else:
        raise ValueError(
            f"Unknown optimizer_type '{optimizer_type}'. "
            "Choose: 'adamw', 'adam', 'sgd', 'rmsprop', 'muonwithadamw', 'adagram', 'adamgram'."
        )

    print(f"[*] Optimizer: {optimizer_type.upper()}  lr={learning_rate:.2e}  wd={weight_decay:.2e}")
    return optimizer


# ---------------------------------------------------------------------------
# LR schedule: cosine with linear warmup
# ---------------------------------------------------------------------------

def get_lr(epoch: int, warmup_epochs: int, max_epochs: int,
           learning_rate: float, min_lr: float) -> float:
    if epoch < warmup_epochs:
        return learning_rate * (epoch + 1) / (warmup_epochs + 1)
    if epoch >= max_epochs:
        return min_lr
    decay_ratio = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    ctx,
    num_classes: int,
) -> dict:
    model.eval()
    total_loss   = 0.0
    top1_correct = 0
    top5_correct = 0
    n_samples    = 0

    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    logits = None
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with ctx:
            logits = model(imgs)
            loss   = F.cross_entropy(logits, labels)

        total_loss   += loss.item() * imgs.size(0)
        n_samples    += imgs.size(0)

        preds = logits.argmax(dim=1)
        top1_correct += (preds == labels).sum().item()

        if logits.size(1) >= 5:
            _, top5 = logits.topk(5, dim=1)
            top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        preds_cpu  = preds.cpu()
        labels_cpu = labels.cpu()
        for c in range(num_classes):
            pred_c  = preds_cpu  == c
            label_c = labels_cpu == c
            tp[c]  += ( pred_c &  label_c).sum()
            fp[c]  += ( pred_c & ~label_c).sum()
            fn[c]  += (~pred_c &  label_c).sum()

    model.train()

    avg_loss = total_loss / n_samples
    top1_acc = 100.0 * top1_correct / n_samples
    top5_acc = (100.0 * top5_correct / n_samples
                if (logits is not None and logits.size(1) >= 5)
                else top1_acc)

    prec_per_class  = tp.float() / (tp + fp).float().clamp(min=1)
    rec_per_class   = tp.float() / (tp + fn).float().clamp(min=1)
    macro_precision = 100.0 * prec_per_class.mean().item()
    macro_recall    = 100.0 * rec_per_class.mean().item()

    return {
        "loss":      avg_loss,
        "top1_acc":  top1_acc,
        "top5_acc":  top5_acc,
        "precision": macro_precision,
        "recall":    macro_recall,
    }


# ---------------------------------------------------------------------------
# CsvLogger
# ---------------------------------------------------------------------------

class ImageCsvLogger:
    def __init__(self, path: str, model_type: str, dataset: str, lr: float):
        import csv
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path   = path
        self.f      = open(path, "w", newline="")
        self.writer = csv.writer(self.f)
        self.writer.writerow([
            "epoch",
            "train_loss", "train_top1", "train_top5",
            "val_loss",   "val_top1",   "val_top5",
            "val_precision", "val_recall",
            "lr", "model_type", "dataset",
        ])
        self.meta = {"model_type": model_type, "dataset": dataset, "lr": lr}

    def log(self, epoch: int, train: dict, val: dict, lr: float):
        self.writer.writerow([
            epoch,
            f"{train['loss']:.6f}",
            f"{train['top1_acc']:.4f}",
            f"{train['top5_acc']:.4f}",
            f"{val['loss']:.6f}",
            f"{val['top1_acc']:.4f}",
            f"{val['top5_acc']:.4f}",
            f"{val['precision']:.4f}",
            f"{val['recall']:.4f}",
            f"{lr:.8f}",
            self.meta["model_type"],
            self.meta["dataset"],
        ])
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global clearml_log, out_dir, dataset, model_type, model_variant
    global finetune_strategy, trainable_stages, trainable_blocks, small_input, dropout
    global batch_size, max_epochs, gradient_accumulation_steps
    global weight_decay, learning_rate, beta1, beta2, grad_clip, optimizer_type, momentum, nesterov
    global seed, device, dtype, compile_model, init_from
    global clearml_project, clearml_task_name, wandb_log
    global augment, eval_interval, always_save_checkpoint
    global decay_lr, warmup_epochs, min_lr, val_fraction, num_workers
    global resize_to

    # ── Config resolution ────────────────────────────────────────────────────
    args = parse_args()
    if args.config:
        apply_config(args.config)
    apply_args(args)

    if wandb_log:
        clearml_log = True

    # ── Distributed init ─────────────────────────────────────────────────────
    # Must happen before any optimizer that calls dist.get_world_size()
    # (e.g. MuonWithAuxAdam). Safe to call unconditionally — the guard inside
    # _init_distributed() prevents double-initialisation.
    _init_distributed()

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(out_dir, exist_ok=True)

    # ── AMP context ──────────────────────────────────────────────────────────
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32,
               "bfloat16": torch.bfloat16,
               "float16": torch.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))

    # ── Dataset meta ─────────────────────────────────────────────────────────
    meta_path = os.path.join("data", dataset, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Dataset metadata not found at '{meta_path}'. "
            f"Run: python data/{dataset}/prepare.py"
        )
    with open(meta_path) as f:
        meta = json.load(f)

    num_classes = meta["num_classes"]
    in_channels = meta["image_size"][0]
    print(f"[*] Dataset: {dataset} — {num_classes} classes, {meta['n_train']:,} train / "
          f"{meta['n_val']:,} val / {meta['n_test']:,} test")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=dataset,
        meta=meta,
        batch_size=batch_size,
        val_fraction=val_fraction,
        augment=augment,
        num_workers=num_workers,
        seed=seed,
        resize_to=resize_to,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_val_acc = -1.0

    if init_from == "resume":
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        print(f"[*] Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model(
            model_type=ckpt.get("model_type", model_type),
            model_variant=ckpt.get("model_variant", model_variant),
            num_classes=num_classes,
            in_channels=in_channels,
            init_from="scratch",
            finetune_strategy=finetune_strategy,
            trainable_stages=trainable_stages,
            trainable_blocks=trainable_blocks,
            small_input=small_input,
            dropout=dropout,
        )
        model.load_state_dict(ckpt["model"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", -1.0)
    else:
        model = build_model(
            model_type=model_type,
            model_variant=model_variant,
            num_classes=num_classes,
            in_channels=in_channels,
            init_from=init_from,
            finetune_strategy=finetune_strategy,
            trainable_stages=trainable_stages,
            trainable_blocks=trainable_blocks,
            small_input=small_input,
            dropout=dropout,
        )

    model = model.to(device)

    if compile_model:
        print("[*] Compiling model with torch.compile()…")
        model = torch.compile(model)

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = build_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        momentum=momentum,
        nesterov=nesterov,
        model_type=model_type,
        rank=rank
    )

    if init_from == "resume" and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    # ── ClearML ───────────────────────────────────────────────────────────────
    clearml_task = None
    if clearml_log:
        try:
            from clearml import Task
            clearml_task = Task.init(project_name=clearml_project, task_name=clearml_task_name)
            clearml_task.connect({
                "dataset": dataset, "model_type": model_type, "model_variant": model_variant,
                "batch_size": batch_size, "max_epochs": max_epochs, "learning_rate": learning_rate,
                "weight_decay": weight_decay, "dropout": dropout, "augment": augment,
                "init_from": init_from, "finetune_strategy": finetune_strategy,
                "seed": seed, "dtype": dtype,
            }, name="hyperparameters")
            print("[✓] ClearML task initialised.")
        except Exception as e:
            print(f"[!] ClearML init failed: {e}. Continuing without remote logging.")
            clearml_task = None
            clearml_log  = False
        if clearml_log and clearml_task is not None:
            logger = clearml_task.get_logger()
        # ── Pre-training baseline (OUTSIDE the clearml block) ─────────────────────
        # ── Pre-training baseline ─────────────────────────────────────────────────
        print("[*] Pre-training evaluation (epoch 0 baseline)…")
        val_metrics_init   = evaluate(model, val_loader,   device, ctx, num_classes)
        train_metrics_init = evaluate(model, train_loader, device, ctx, num_classes)
        print(f"    init train_loss={train_metrics_init['loss']:.4f} | "
              f"val_loss={val_metrics_init['loss']:.4f} | "
              f"val_top1={val_metrics_init['top1_acc']:.2f}%")

        if clearml_log and clearml_task is not None:
            _logger = clearml_task.get_logger()
            _logger.report_scalar("loss",      "train", train_metrics_init["loss"],      0)
            _logger.report_scalar("loss",      "val",   val_metrics_init["loss"],        0)
            _logger.report_scalar("top1_acc",  "train", train_metrics_init["top1_acc"],  0)
            _logger.report_scalar("top1_acc",  "val",   val_metrics_init["top1_acc"],    0)
            _logger.report_scalar("top5_acc",  "val",   val_metrics_init["top5_acc"],    0)
            _logger.report_scalar("precision", "val",   val_metrics_init["precision"],   0)
            _logger.report_scalar("recall",    "val",   val_metrics_init["recall"],      0)
            _logger.report_scalar("lr",        "lr",    learning_rate,                   0)
    # # ── CSV logger ────────────────────────────────────────────────────────────
    # csv_path = os.path.join(out_dir, f"log_{dataset}_{model_type}_{model_variant}.csv")
    # csv_logger = ImageCsvLogger(
    #     path=csv_path,
    #     model_type=f"{model_type}/{model_variant}",
    #     dataset=dataset,
    #     lr=learning_rate,
    # )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training {model_type}/{model_variant} on {dataset}")
    print(f"  Device: {device}  |  dtype: {dtype}  |  batch: {batch_size}")
    print(f"  Epochs: {max_epochs}  |  LR: {learning_rate:.2e}")
    print(f"{'='*60}\n")

    steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, max_epochs):
        model.train()

        lr = get_lr(epoch, warmup_epochs, max_epochs, learning_rate, min_lr) if decay_lr else learning_rate
        for pg in optimizer.param_groups:
            if "_base_lr" not in pg:
                pg["_base_lr"] = pg["lr"]
            pg["lr"] = pg["_base_lr"] * (lr / learning_rate)

        epoch_loss = 0.0
        epoch_top1 = 0
        epoch_n    = 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        for step, (imgs, labels) in enumerate(train_loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with ctx:
                logits = model(imgs)
                loss   = F.cross_entropy(logits, labels)
                loss   = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_loss  = loss.item() * gradient_accumulation_steps
            epoch_loss += batch_loss * imgs.size(0)
            epoch_top1 += (logits.detach().argmax(1) == labels).sum().item()
            epoch_n    += imgs.size(0)

            if step % log_interval == 0:
                elapsed = time.time() - t0
                print(
                    f"  epoch {epoch+1:3d}/{max_epochs} | "
                    f"step {step+1:4d}/{steps_per_epoch} | "
                    f"loss {batch_loss:.4f} | "
                    f"acc {100.*epoch_top1/epoch_n:.2f}% | "
                    f"lr {lr:.2e} | "
                    f"{elapsed:.1f}s"
                )

        train_metrics = {
            "loss":     epoch_loss / epoch_n,
            "top1_acc": 100.0 * epoch_top1 / epoch_n,
            "top5_acc": -1.0,
        }

        # ── Evaluation ────────────────────────────────────────────────────────
        if (epoch + 1) % eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device, ctx, num_classes)
            epoch_time  = time.time() - t0

            print(
                f"\n[Epoch {epoch+1:3d}/{max_epochs}] "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_top1={val_metrics['top1_acc']:.2f}% | "
                f"val_top5={val_metrics['top5_acc']:.2f}% | "
                f"val_prec={val_metrics['precision']:.2f}% | "
                f"val_rec={val_metrics['recall']:.2f}% | "
                f"time={epoch_time:.1f}s\n"
            )

            if clearml_log and clearml_task is not None:
                logger = clearml_task.get_logger()
                try:
                    logger.report_scalar("loss",      "train", train_metrics["loss"],      epoch+1)
                    logger.report_scalar("loss",      "val",   val_metrics["loss"],        epoch+1)
                    logger.report_scalar("top1_acc",  "train", train_metrics["top1_acc"],  epoch+1)
                    logger.report_scalar("top1_acc",  "val",   val_metrics["top1_acc"],    epoch+1)
                    logger.report_scalar("top5_acc",  "val",   val_metrics["top5_acc"],    epoch+1)
                    logger.report_scalar("precision", "val",   val_metrics["precision"],   epoch+1)
                    logger.report_scalar("recall",    "val",   val_metrics["recall"],      epoch+1)
                    logger.report_scalar("lr",        "lr",    lr,                         epoch+1)
                except Exception as e:
                    print(f"[!] ClearML logging error: {e}")

            # csv_logger.log(epoch, train_metrics, val_metrics, lr)

            improved = val_metrics["top1_acc"] > best_val_acc
            if improved:
                best_val_acc = val_metrics["top1_acc"]
            # if improved or always_save_checkpoint:
                # ckpt = {
                #     "model":         (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                #     "optimizer":     optimizer.state_dict(),
                #     "epoch":         epoch,
                #     "best_val_acc":  best_val_acc,
                #     "model_type":    model_type,
                #     "model_variant": model_variant,
                #     "dataset":       dataset,
                #     "config": {
                #         k: globals()[k] for k in [
                #             "dataset", "model_type", "model_variant",
                #             "batch_size", "max_epochs", "learning_rate",
                #             "weight_decay", "dropout", "augment", "seed",
                #         ]
                #     },
                # }
                # ckpt_path = os.path.join(out_dir, "ckpt.pt")
                # torch.save(ckpt, ckpt_path)
                # marker = " ← best" if improved else ""
                # print(f"[✓] Checkpoint saved ({ckpt_path}){marker}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n[*] Final test evaluation…")
    best_ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(best_ckpt["model"])
        print(f"    Loaded best checkpoint (epoch {best_ckpt['epoch']+1}, "
              f"val_top1={best_ckpt['best_val_acc']:.2f}%)")

    test_metrics = evaluate(model, test_loader, device, ctx, num_classes)
    print(
        f"\n{'='*60}\n"
        f"  TEST RESULTS\n"
        f"  test_loss      = {test_metrics['loss']:.4f}\n"
        f"  test_top1      = {test_metrics['top1_acc']:.2f}%\n"
        f"  test_top5      = {test_metrics['top5_acc']:.2f}%\n"
        f"  test_precision = {test_metrics['precision']:.2f}%\n"
        f"  test_recall    = {test_metrics['recall']:.2f}%\n"
        f"{'='*60}\n"
    )

    if clearml_log and clearml_task is not None:
        logger = clearml_task.get_logger()
        try:
            logger.report_single_value("test_top1_acc",  test_metrics["top1_acc"])
            logger.report_single_value("test_loss",      test_metrics["loss"])
            logger.report_single_value("test_precision", test_metrics["precision"])
            logger.report_single_value("test_recall",    test_metrics["recall"])
        except Exception as e:
            print(f"[!] ClearML final logging error: {e}")
        clearml_task.close()

    # csv_logger.close()
    # print(f"[✓] CSV log saved to {csv_path}")
    print("[✓] Done.")


if __name__ == "__main__":
    main()