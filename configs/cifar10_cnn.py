# ---------------------------------------------------------------------------
# Config: ConvNet (medium) on CIFAR-10 — trained from scratch
#
# Usage:
#   python data/cifar10/prepare.py         # one-time dataset prep
#   python train_image.py --config config/cifar10_cnn.py
#   python train_image.py --config config/cifar10_cnn.py --clearml_log=True
# ---------------------------------------------------------------------------

# Dataset
dataset       = "cifar10"
val_fraction  = 0.1
num_workers   = 4
augment       = True   # RandomCrop, RandomHFlip, ColorJitter

# Model — BN-equipped ConvNet with 64→128→256 channels
model_type    = "cnn"
model_variant = "medium"   # 'small' | 'medium' | 'large'
dropout       = 0.2

# Initialisation
init_from     = "scratch"

# Training
batch_size    = 128
max_epochs    = 60
gradient_accumulation_steps = 8

# Optimiser (AdamW)
learning_rate = 1e-3
weight_decay  = 5e-4
beta1         = 0.9
beta2         = 0.999
grad_clip     = 1.0

# LR schedule (cosine with linear warmup)
decay_lr      = True
warmup_epochs = 5
min_lr        = 1e-5

# Logging
eval_interval          = 1
always_save_checkpoint = False
# out_dir                = "out-cifar10-cnn-medium"

# ClearML (disabled by default; pass --clearml_log=True to enable)
clearml_log        = True
clearml_project    = "cnn-image"
# clearml_task_name  = "cifar10-cnn-medium"

# System
seed          = 42
device        = "cuda"
dtype         = "bfloat16"
compile_model = False
