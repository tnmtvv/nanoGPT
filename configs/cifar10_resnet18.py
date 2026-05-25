# ---------------------------------------------------------------------------
# Config: ResNet-18 (pretrained on ImageNet) fine-tuned on CIFAR-10
#
# Usage:
#   python data/cifar10/prepare.py         # one-time dataset prep
#   python train_image.py --config config/cifar10_resnet18.py
#   python train_image.py --config config/cifar10_resnet18.py --clearml_log=True
#
# Notes:
#   - `init_from = "pretrained"` loads ImageNet weights from torchvision.
#   - `small_input = True` replaces the 7×7 stride-2 conv with a 3×3 stride-1
#     conv so that 32×32 CIFAR images are not immediately over-downsampled.
#   - `finetune_strategy = "full"` unlocks all layers.  Switch to "last_n"
#     with `trainable_stages = 2` for a lighter partial fine-tune.
#   - Differential LR: backbone LR = learning_rate, head LR = learning_rate×10.
# ---------------------------------------------------------------------------

# Dataset
dataset       = "cifar10"
val_fraction  = 0.1
num_workers   = 4
augment       = True

# Model
model_type        = "resnet"
model_variant     = "resnet18"
finetune_strategy = "full"    # 'full' | 'head_only' | 'last_n'
trainable_stages  = 2         # used only when finetune_strategy == 'last_n'
small_input       = True      # adapt first conv for 32×32 inputs
dropout           = 0.0

# Initialisation — load ImageNet pretrained weights
init_from     = "pretrained"

# Training
batch_size    = 128
max_epochs    = 30
gradient_accumulation_steps = 1

# Optimiser (AdamW with differential LR)
learning_rate = 1e-3    # backbone LR; head gets ×10
weight_decay  = 1e-4
beta1         = 0.9
beta2         = 0.999
grad_clip     = 1.0

# LR schedule
decay_lr      = True
warmup_epochs = 3
min_lr        = 1e-6

# Logging
eval_interval          = 1
always_save_checkpoint = False
out_dir                = "out-cifar10-resnet18"

# ClearML (disabled by default; pass --clearml_log=True to enable)
clearml_log        = True
clearml_project    = "resnet18-image"
clearml_task_name  = "cifar10-resnet18-pretrained"

# System
seed          = 42
device        = "cuda"
dtype         = "bfloat16"
compile_model = False
