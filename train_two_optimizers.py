"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import sys 
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

site_packages_path = '/opt/miniconda3/envs/nanogpt_python39/lib/python3.9/site-packages'

# Add this path to the Python's search paths if it's not already there
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)

# from adagram_optimizers import AdamAdagram


from model import GPTConfig, GPT
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description='Train a GPT model')

# I/O arguments
parser.add_argument('--config', type=str, default='config', help='Config')
parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
parser.add_argument('--eval_interval', type=int, default=2000, help='Evaluation interval')
parser.add_argument('--log_interval', type=int, default=1, help='Logging interval')
parser.add_argument('--eval_iters', type=int, default=200, help='Number of evaluation iterations')
parser.add_argument('--eval_only', action='store_true', help='Exit after first eval')
parser.add_argument('--always_save_checkpoint', action='store_true', help='Always save checkpoint after eval')
parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], help='Initialize from scratch, resume, or pretrained')

# Wandb logging
parser.add_argument('--wandb_log', action='store_true', help='Enable wandb logging')
parser.add_argument('--wandb_project', type=str, default='owt', help='Wandb project name')
parser.add_argument('--wandb_run_name', type=str, default='gpt2', help='Wandb run name')

# Data arguments
parser.add_argument('--dataset', type=str, default='openwebtext', help='Dataset name')
parser.add_argument('--gradient_accumulation_steps', type=int, default=40, help='Gradient accumulation steps')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--block_size', type=int, default=1024, help='Block size')

# Model arguments
parser.add_argument('--n_layer', type=int, default=12, help='Number of layers')
parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--bias', action='store_true', help='Use bias in LayerNorm and Linear layers')
parser.add_argument('--seed', type=float, default=0.0, help='seed')

# Optimizer arguments
parser.add_argument('--optimizer_name', type=str, default="AdamW", help='optimizer choice')
parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
parser.add_argument('--rank', type=float, default=1, help='optimizer rank')
parser.add_argument('--max_iters', type=int, default=600000, help='Maximum iterations')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
# parser.add_argument('--rank', type=int, default=1, help='Rank for low-rank optimizers')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')

# Learning rate decay
parser.add_argument('--decay_lr', action='store_true', default=True, help='Decay learning rate')
parser.add_argument('--warmup_iters', type=int, default=2000, help='Warmup iterations')
parser.add_argument('--lr_decay_iters', type=int, default=600000, help='LR decay iterations')
parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')

# DDP arguments
parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='DDP backend')

# System arguments
parser.add_argument('--device', type=str, default='cuda', help='Device (cuda, cpu, etc.)')
parser.add_argument('--dtype', type=str, default='float16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', 
                    choices=['float32', 'bfloat16', 'float16'], help='Data type')
parser.add_argument('--compile', action='store_true', default=True, help='Compile model with PyTorch 2.0')
# Parse arguments
args = parser.parse_args()


# Update global variables from parsed arguments
out_dir = args.out_dir
eval_interval = args.eval_interval
log_interval = args.log_interval
eval_iters = args.eval_iters
eval_only = args.eval_only
always_save_checkpoint = args.always_save_checkpoint
init_from = args.init_from
wandb_log = args.wandb_log
wandb_project = args.wandb_project
wandb_run_name = args.wandb_run_name
dataset = args.dataset
gradient_accumulation_steps = args.gradient_accumulation_steps
batch_size = args.batch_size
block_size = args.block_size
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
dropout = args.dropout
bias = args.bias
learning_rate = args.learning_rate
max_iters = args.max_iters
weight_decay = args.weight_decay
beta1 = args.beta1
beta2 = args.beta2
rank = args.rank
grad_clip = args.grad_clip
decay_lr = args.decay_lr
warmup_iters = args.warmup_iters
lr_decay_iters = max_iters
min_lr = args.min_lr
backend = args.backend
device = args.device
dtype = args.dtype
compile = args.compile
optimizer_name = args.optimizer_name
seed = args.seed
rank = args.rank
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
ddp = False
if ddp:
    init_process_group(backend=backend, world_size=1, rank=0)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    dist.init_process_group(backend='nccl', init_method='env://')
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer_adagram, optimizer_adamw = model.configure_two_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, rank)
if init_from == 'resume':
    optimizer_adagram.load_state_dict(checkpoint['optimizer_adagram'])
    optimizer_adamw.load_state_dict(checkpoint['optimizer_adamw'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if master_process:
    print("clearml init")
    from clearml import Task
    task = Task.init(project_name=wandb_project, task_name=wandb_run_name)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # 1. Determine and set the learning rate for this iteration for BOTH optimizers
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer_adagram.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_adamw.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print("logging")
        if master_process:
            task.get_logger().report_scalar("train_loss", "loss", losses['train'], iter_num)
            task.get_logger().report_scalar("val_loss", "loss", losses['val'], iter_num)
            task.get_logger().report_scalar("lr", "lr", lr, iter_num)
            task.get_logger().report_scalar("learning_rate", "lr", lr, iter_num)
            task.get_logger().report_scalar("model_flops_utilization", "mfu_percent", running_mfu*100, iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            # if iter_num > 0:
                # 2. Save state dictionaries for BOTH optimizers
                # checkpoint = {
                #     'model': raw_model.state_dict(),
                #     'optimizer_adagram': optimizer_adagram.state_dict(),
                #     'optimizer_adamw': optimizer_adamw.state_dict(),
                #     'model_args': model_args,
                #     'iter_num': iter_num,
                #     'best_val_loss': best_val_loss,
                #     'config': config,
                # }
                # print(f"saving checkpoint to {out_dir}")
                # torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    if iter_num == 0 and eval_only:
        break

    # Forward backward update
    for micro_step in range(gradient_accumulation_steps):
        # (DDP and context management code remains the same)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    # Clip the gradient
    if grad_clip != 0.0:
        # Unscale gradients for both optimizers before clipping
        scaler.unscale_(optimizer_adagram)
        scaler.unscale_(optimizer_adamw)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # 3. Step BOTH optimizers and scaler
    scaler.step(optimizer_adagram)
    scaler.step(optimizer_adamw)
    scaler.update()

    # 4. Flush the gradients for BOTH optimizers
    optimizer_adagram.zero_grad(set_to_none=True)
    optimizer_adamw.zero_grad(set_to_none=True)

    # (Timing and logging code remains the same)
    
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        print("end")
        break

if master_process:
    task.close()
    destroy_process_group()

if ddp:
    destroy_process_group()
