# ---------------------------------------------------------------------------
# Config: ResNet-50 (pretrained on ImageNet) fine-tuned on CIFAR-100
#
# Usage:
#   python data/cifar100/prepare.py        # one-time dataset prep
#   python train_image.py --config config/cifar100_resnet50.py
#   python train_image.py --config config/cifar100_resnet50.py --clearml_log=True
#
# Notes:
#   - CIFAR-100 has 100 classes — ResNet-50 is a good capacity fit.
#   - `small_input = True` adapts the first conv for 32×32 images.
#   - `finetune_strategy = "last_n"` with `trainable_stages = 3` unfreezes
#     layer2–layer4 + head; cheaper than full fine-tuning while still
#     adapting mid/high-level features.  Switch to "full" for best accuracy.
# ---------------------------------------------------------------------------

# Dataset
dataset       = "cifar100"
val_fraction  = 0.1
num_workers   = 4
augment       = True

# Model
model_type        = "resnet"
model_variant     = "resnet50"
finetune_strategy = "last_n"    # 'full' | 'head_only' | 'last_n'
trainable_stages  = 3           # unfreeze layer4, layer3, layer2 + head
small_input       = True
dropout           = 0.2         # some regularisation for 100-way classification

# Initialisation
init_from     = "pretrained"

# Training
batch_size    = 128
max_epochs    = 40
gradient_accumulation_steps = 1

# Optimiser
learning_rate = 5e-4    # backbone LR; head gets ×10 = 5e-3
weight_decay  = 1e-4
beta1         = 0.9
beta2         = 0.999
grad_clip     = 1.0

# LR schedule
decay_lr      = True
warmup_epochs = 5
min_lr        = 1e-6

# Logging
eval_interval          = 1
always_save_checkpoint = False
out_dir                = "out-cifar100-resnet50"

# ClearML (disabled by default; pass --clearml_log=True to enable)
clearml_log        = False
clearml_project    = "nanoGPT-image"
clearml_task_name  = "cifar100-resnet50-pretrained"

# System
seed          = 42
device        = "cuda"
dtype         = "bfloat16"
compile_model = False
