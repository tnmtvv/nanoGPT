# ---------------------------------------------------------------------------
# Config: LeNet-5 on MNIST
#
# Usage:
#   python data/mnist/prepare.py          # one-time dataset prep
#   python train_image.py --config config/mnist_cnn.py
#   python train_image.py --config config/mnist_cnn.py --clearml_log=True
# ---------------------------------------------------------------------------

# Dataset
dataset       = "mnist"
val_fraction  = 0.1
num_workers   = 4
augment       = True   # RandomCrop(28, padding=4)

# Model
model_type    = "cnn"
model_variant = "lenet5"   # LeNet-5 architecture
dropout       = 0.2

# Initialisation
init_from     = "scratch"

# Training
batch_size    = 128
max_epochs    = 20
gradient_accumulation_steps = 1

# Optimiser (AdamW)
learning_rate = 1e-3
weight_decay  = 1e-4
beta1         = 0.9
beta2         = 0.999
grad_clip     = 1.0

# LR schedule (cosine with linear warmup)
decay_lr      = True
warmup_epochs = 3
min_lr        = 1e-5

# Logging
eval_interval          = 1       # evaluate every epoch
always_save_checkpoint = False   # only save on val_acc improvement
out_dir                = "out-mnist-lenet5"

# ClearML (disabled by default; pass --clearml_log=True to enable)
clearml_log        = False
clearml_project    = "nanoGPT-image"
clearml_task_name  = "mnist-lenet5"

# System
seed          = 42
device        = "cuda"   # falls back to cpu automatically in train_image.py
dtype         = "float32"   # MNIST is small enough for fp32
compile_model = False
