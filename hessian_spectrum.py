import torch
import tiktoken
from model import GPT, GPTConfig
import scipy.sparse.linalg as spsplin

def hessian_vector_product(model, input_ids, targets, vector):
    """Compute Hessian-vector product."""
    model.zero_grad()
    
    # Additional safety check
    vocab_size = model.transformer.wte.weight.shape[0]
    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    targets = torch.clamp(targets, 0, vocab_size - 1)
    
    logits, loss = model(input_ids, targets)
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad])
    dot_product = torch.dot(grad_vec, vector)
    
    hvp = torch.autograd.grad(dot_product, model.parameters())
    hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
    return hvp_flat

# Sample texts
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning transforms artificial intelligence research.",
    "Deep neural networks learn complex patterns from data.",
    "Natural language processing enables text understanding.",
    "Large language models require extensive computational resources.",
] * 20

device = "cuda"

# Load model FIRST
print("Loading checkpoint...")
checkpoint = torch.load("large_bs_checkpoints/2000_ckpt.pt", 
                       map_location="cpu", weights_only=False)

model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)

if hasattr(gptconf, 'flash'):
    gptconf.flash = False


model = GPT(gptconf)

state_dict = checkpoint['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k.replace('_orig_mod.', '')] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()

# Get ACTUAL vocab size
vocab_size = model.transformer.wte.weight.shape[0]
block_size = model.config.block_size

print(f"Model embedding vocabulary: {vocab_size}")
print(f"Model block size: {block_size}")

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
print(f"Tokenizer vocabulary: {enc.n_vocab}")

# Check mismatch
if vocab_size != enc.n_vocab:
    print(f"\n⚠️  MISMATCH DETECTED!")
    print(f"Model has {vocab_size} embeddings, tokenizer has {enc.n_vocab} tokens")

# Tokenize with proper clamping
max_length = min(128, block_size)
input_ids_list = []

print(f"\nTokenizing {len(sample_texts)} texts...")
for text in sample_texts:
    # Clean and encode
    text = " ".join(text.split())
    tokens = enc.encode(text)
    
    # Truncate
    tokens = tokens[:max_length]
    
    # *** CRITICAL FIX: Clip tokens to model's vocab ***
    tokens = [min(max(t, 0), vocab_size - 1) for t in tokens]
    
    if len(tokens) > 0:
        input_ids_list.append(tokens)

print(f"Processed {len(input_ids_list)} sequences")

# Pad
max_len = max(len(tokens) for tokens in input_ids_list)
padded_input_ids = [tokens + [0] * (max_len - len(tokens)) 
                    for tokens in input_ids_list]

# To tensor
input_ids = torch.tensor(padded_input_ids, dtype=torch.long)

# VERIFY
print(f"\nPre-GPU Validation:")
print(f"  Shape: {input_ids.shape}")
print(f"  Min: {input_ids.min().item()}, Max: {input_ids.max().item()}")
print(f"  Valid: [0, {vocab_size-1}]")

if input_ids.max().item() >= vocab_size:
    raise ValueError(f"Token {input_ids.max().item()} >= vocab {vocab_size}")

# Move to GPU
print("\nMoving to GPU...")
model.to(device)
input_ids = input_ids.to(device)
targets = input_ids.clone()

# Test forward
print("Testing forward pass...")
with torch.no_grad():
    logits, loss = model(input_ids, targets)
print(f"✓ Forward pass successful! Loss: {loss.item():.4f}")

# Hessian computation
num_params = sum(p.numel() for p in model.parameters())
print(f"\nComputing Hessian for {num_params:,} parameters...")

def hvp_numpy(x):
    x_torch = torch.tensor(x, dtype=torch.float32).to(device)
    return hessian_vector_product(model, input_ids, targets, x_torch).cpu().numpy()

A = spsplin.LinearOperator((num_params, num_params), matvec=hvp_numpy)

print("Computing eigenvalues...")
res_max = spsplin.eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-3)
print(f"Max eigenvalue: {res_max[0]:.6f}")

res_min = spsplin.eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-3)
print(f"Min eigenvalue: {res_min[0]:.6f}")

print(f"\nCondition number: {abs(res_max[0]/res_min[0]):.2f}")
