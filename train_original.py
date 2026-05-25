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
from CsvLogger import CsvLogger
from metrics import compute_perplexity, compute_bits_per_character, compute_token_accuracy, compute_top_k_accuracy, compute_confidence_metrics

site_packages_path = '/opt/miniconda3/envs/nanogpt_python39/lib/python3.9/site-packages'

# Add this path to the Python's search paths if it's not already there
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)

from adagram_optimizers.AdagramPS import AdaGramPS


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
parser.add_argument('--scheduler', type=str, default='cos', help='Scheduler type')
parser.add_argument('--rank', type=float, default=1, help='optimizer rank')
parser.add_argument('--alpha', type=float, default=1, help='optimizer alpha')
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
scheduler = args.scheduler
max_iters = args.max_iters
weight_decay = args.weight_decay
beta1 = args.beta1
beta2 = args.beta2
rank = args.rank
alpha = args.alpha
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

wsd_cooldown_frac = 0.2
wsd_decay_type = "cosine"



# # -----------------------------------------------------------------------------
# # default config values designed to train a gpt2 (124M) on OpenWebText
# # I/O
# out_dir = 'out'
# eval_interval = 2000
# log_interval = 1
# eval_iters = 200
# eval_only = False # if True, script exits right after the first eval
# always_save_checkpoint = False # if True, always save a checkpoint after each eval
# init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# # wandb logging
# wandb_log = False # disabled by default
# wandb_project = 'owt'
# wandb_run_name = 'gpt2' # 'run' + str(time.time())
# # data
# dataset = 'openwebtext'
# gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
# batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
# block_size = 1024 
# # model
# n_layer = 12
# n_head = 12
# n_embd = 768
# dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# bias = False # do we use bias inside LayerNorm and Linear layers?
# # adamw optimizer
# learning_rate = 6e-4 # max learning rate
# max_iters = 600000 # total number of training iterations
# weight_decay = 1e-1
# beta1 = 0.9
# beta2 = 0.95
# rank = 1
# grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# # learning rate decay settings
# decay_lr = True # whether to decay the learning rate
# warmup_iters = 2000 # how many steps to warm up for
# lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
# min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# # DDP settings
# backend = 'nccl' # 'nccl', 'gloo', etc.
# # system
# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# compile = True # use PyTorch 2.0 to compile the model to be faster
# # -----------------------------------------------------------------------------
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
    os.environ['MASTER_PORT'] = '29501'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    # dist.init_process_group(backend='nccl', init_method='env://')
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset + seed)
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
print("configure optimizer")
print()
print()
print()

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, opt_name=optimizer_name, rank=int(rank), alpha=alpha)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
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

@torch.no_grad()
def estimate_loss_with_metrics(model, ctx):
    """
    Enhanced evaluation function that computes loss and additional metrics.
    """
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        perplexities = torch.zeros(eval_iters)
        token_accuracies = torch.zeros(eval_iters)
        top5_accuracies = torch.zeros(eval_iters)
        confidences = torch.zeros(eval_iters)
        entropies = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            
            # Basic metrics
            losses[k] = loss.item()
            perplexities[k] = compute_perplexity(loss)
            
            # Accuracy metrics
            token_accuracies[k] = compute_token_accuracy(logits, Y)
            top5_accuracies[k] = compute_top_k_accuracy(logits, Y, k=5)
            
            # Confidence metrics
            mean_conf, entropy = compute_confidence_metrics(logits)
            confidences[k] = mean_conf
            entropies[k] = entropy
        
        # Store all metrics
        out[split] = {
            'loss': losses.mean().item(),
            'perplexity': perplexities.mean().item(),
            'token_accuracy': token_accuracies.mean().item(),
            'top5_accuracy': top5_accuracies.mean().item(),
            'mean_confidence': confidences.mean().item(),
            'entropy': entropies.mean().item(),
            'bpc': compute_bits_per_character(losses.mean())
        }
    
    model.train()
    return out

# ── Cosine with warmup (nanoGPT default, unchanged) ─────────────────────────
def get_lr_cos(it):
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


# ── Warmup-Stable-Decay (WSD / Trapezoidal) ─────────────────────────────────
def get_lr_wsd(it):
    _decay_steps = int(lr_decay_iters * wsd_cooldown_frac)
    _cooldown_start = lr_decay_iters - _decay_steps   # step where plateau ends

    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it < _cooldown_start:                           # flat plateau
        return learning_rate
    if it <= lr_decay_iters:                           # cooldown
        progress = (it - _cooldown_start) / _decay_steps   # 0 → 1
        if wsd_decay_type == "cosine":
            return min_lr + 0.5 * (learning_rate - min_lr) * (1.0 + math.cos(math.pi * progress))
        return learning_rate - (learning_rate - min_lr) * progress   # linear
    return min_lr


# ── Linear decay ─────────────────────────────────────────────────────────────
def get_lr_linear(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it >= lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + (learning_rate - min_lr) * (1.0 - decay_ratio)


# logging
if master_process and wandb_log ==  True:
    print("clearml init")
    from clearml import Task
    task = Task.init(project_name=wandb_project, task_name=wandb_run_name)
print("optimizer_name", optimizer_name)

# csv_logger = CsvLogger(opt_name=optimizer_name, bs=batch_size, lr=learning_rate, filename=f"out-{optimizer_name}-bs{batch_size}-lr{learning_rate}/log.csv", seed=seed)
csv_logger = CsvLogger(opt_name=optimizer_name, bs=batch_size, lr=learning_rate, filename=f"out-{optimizer_name}-check/log.csv", seed=seed)

# training loop
X, Y = get_batch('train', ) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
print("!!!")
print(f"scheduler {scheduler}")
while True:

    # determine and set the learning rate for this iteration
    if scheduler == 'cos':
        lr = get_lr_cos(iter_num) 
    elif scheduler == 'linear':
        lr = get_lr_linear(iter_num)
    elif scheduler == 'wsd':
        lr = get_lr_wsd(iter_num) 
    elif scheduler == 'no':
        lr = learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        metrics = estimate_loss_with_metrics(model, ctx)

        if wandb_log:
            # FIXED: Access nested dictionary values
            print(f"step {iter_num}: train loss {metrics['train']['loss']:.4f}, val loss {metrics['val']['loss']:.4f}")
            print(f"  train ppl: {metrics['train']['perplexity']:.4f}, val ppl: {metrics['val']['perplexity']:.4f}")
            print("logging")

            if master_process:
                # Loss metrics
                task.get_logger().report_scalar("loss", "train", metrics['train']['loss'], iter_num)
                task.get_logger().report_scalar("loss", "val", metrics['val']['loss'], iter_num)

                # Perplexity metrics - FIXED
                try:
                    task.get_logger().report_scalar("perplexity", "train", metrics['train']['perplexity'], iter_num)
                    task.get_logger().report_scalar("perplexity", "val", metrics['val']['perplexity'], iter_num)
                except Exception as e:
                    print(f"ERROR logging perplexity: {e}")

                try:
                    task.get_logger().report_scalar("token_accuracy", "train", metrics['train']['token_accuracy'], iter_num)
                    task.get_logger().report_scalar("token_accuracy", "val", metrics['val']['token_accuracy'], iter_num)
                except Exception as e:
                    print(f"ERROR logging token_accuracy: {e}")

                try:
                    task.get_logger().report_scalar("top5_accuracy", "train", metrics['train']['top5_accuracy'], iter_num)
                    task.get_logger().report_scalar("top5_accuracy", "val", metrics['val']['top5_accuracy'], iter_num)
                except Exception as e:
                    print(f"ERROR logging top5_accuracy: {e}")

                    # Confidence - FIXED
                    task.get_logger().report_scalar("confidence", "train", metrics['train']['mean_confidence'], iter_num)
                    task.get_logger().report_scalar("confidence", "val", metrics['val']['mean_confidence'], iter_num)

                    # Entropy - FIXED
                    task.get_logger().report_scalar("entropy", "train", metrics['train']['entropy'], iter_num)
                    task.get_logger().report_scalar("entropy", "val", metrics['val']['entropy'], iter_num)

                    # Bits per character - FIXED
                    task.get_logger().report_scalar("bits_per_char", "train", metrics['train']['bpc'], iter_num)
                    task.get_logger().report_scalar("bits_per_char", "val", metrics['val']['bpc'], iter_num)

                    # Learning rate and MFU
                    task.get_logger().report_scalar("learning_rate", "lr", lr, iter_num)
                    task.get_logger().report_scalar("model_flops_utilization", "mfu_percent", running_mfu*100, iter_num)


            # Remove the redundant second part
        csv_logger.report_scalar("train", "loss", metrics['train']['loss'], iter_num)
        csv_logger.report_scalar("val", "loss", metrics['val']['loss'], iter_num)
        csv_logger.report_scalar("learning_rate", "lr", lr, iter_num)
        csv_logger.report_scalar("model_flops_utilization", "mfu_percent", running_mfu*100, iter_num)

        if (metrics['val']['loss'] < best_val_loss) or always_save_checkpoint:
            best_val_loss = metrics['val']['loss']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if master_process:
    task.close()

if ddp:
    destroy_process_group()
