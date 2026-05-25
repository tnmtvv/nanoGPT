# ---------------------------------------------------------------------------
# Config: ViT-B/16 (pretrained on ImageNet) fine-tuned on CIFAR-10
#
# Usage:
#   python data/cifar10/prepare.py         # one-time dataset prep
#   python train_image.py --config config/cifar10_vit_b16.py
#   python train_image.py --config config/cifar10_vit_b16.py --clearml_log=True
#
# Notes:
#   - `init_from = "pretrained"` loads ImageNet weights from torchvision.
#   - `resize_to = 224` обязателен для vit_b_16: модель ожидает 224×224,
#     CIFAR-10 даёт 32×32 — скрипт автоматически resize-ит в DataLoader.
#   - `finetune_strategy = "full"` разморозит все блоки.
#     Switch to "last_n" + `trainable_blocks = 4` для лёгкого файнтюна.
#   - Differential LR: backbone LR = learning_rate, head LR = learning_rate×10.
#   - ViT требует больше памяти, чем ResNet-18 — уменьши batch_size если OOM.
# ---------------------------------------------------------------------------


# Dataset
dataset       = "cifar10"
val_fraction  = 0.1
num_workers   = 4
augment       = True


# Model
model_type        = "vit"
model_variant     = "vit_b_16"
finetune_strategy = "last_n"    # 'full' | 'head_only' | 'last_n'
trainable_blocks  = 4         # used only when finetune_strategy == 'last_n'
dropout           = 0.0


# ViT требует вход 224×224 — resize происходит в DataLoader
resize_to     = 224
small_input   = False         # не используется для ViT


# Initialisation — load ImageNet pretrained weights
init_from     = "pretrained"


# Training
batch_size    = 64            # ViT-B/16 тяжелее ResNet-18, уменьшен с 128
max_epochs    = 30
gradient_accumulation_steps = 2  # эффективный батч = 128


# Optimiser (AdamW с differential LR)
learning_rate = 1e-4    # для ViT лучше меньший LR чем у ResNet
weight_decay  = 1e-4
beta1         = 0.9
beta2         = 0.999
grad_clip     = 1.0


# LR schedule
decay_lr      = True
warmup_epochs = 5             # ViT чувствителен к warmup, увеличен с 3
min_lr        = 1e-6


# Logging
eval_interval          = 1
always_save_checkpoint = False
out_dir                = "out-cifar10-vit-b16"


# ClearML
clearml_log        = True
clearml_project    = "vit-image"
clearml_task_name  = "cifar10-vit-b16-pretrained"


# System
seed          = 42
device        = "cuda"
dtype         = "bfloat16"
compile_model = False