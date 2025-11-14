import torch
import tiktoken
from model import GPT, GPTConfig
import scipy.sparse.linalg as spsplin
import pandas as pd
import glob
import re
import os


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


def extract_iter_num(checkpoint_path):
    """Extract iteration number from checkpoint filename."""
    filename = os.path.basename(checkpoint_path)
    # Extract the first number before underscore
    match = re.search(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract iteration number from {filename}")


def compute_eigenvalues(checkpoint_path, input_ids, targets, device="cuda"):
    """Compute min and max eigenvalues for a given checkpoint."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    
    if hasattr(gptconf, 'flash'):
        gptconf.flash = False
    
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k.replace('_orig_mod.', '')] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Move data to device
    input_ids_gpu = input_ids.to(device)
    targets_gpu = targets.to(device)
    
    # Test forward pass
    with torch.no_grad():
        logits, loss = model(input_ids_gpu, targets_gpu)
    print(f"✓ Forward pass successful! Loss: {loss.item():.4f}")
    
    # Compute eigenvalues
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Computing Hessian for {num_params:,} parameters...")
    
    def hvp_numpy(x):
        x_torch = torch.tensor(x, dtype=torch.float32).to(device)
        return hessian_vector_product(model, input_ids_gpu, targets_gpu, x_torch).cpu().numpy()
    
    A = spsplin.LinearOperator((num_params, num_params), matvec=hvp_numpy)
    
    print("Computing max eigenvalue...")
    res_max = spsplin.eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-3)
    max_sv = res_max[0]
    print(f"Max eigenvalue: {max_sv:.6f}")
    
    print("Computing min eigenvalue...")
    res_min = spsplin.eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-3)
    min_sv = res_min[0]
    print(f"Min eigenvalue: {min_sv:.6f}")
    
    print(f"Condition number: {abs(max_sv/min_sv):.2f}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return min_sv, max_sv


def process_checkpoints(checkpoint_pattern, output_csv="hessian_results.csv", device="cuda"):
    """Process multiple checkpoints and save results to DataFrame."""
    
    # Sample texts (same as before)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning transforms artificial intelligence research.",
        "Deep neural networks learn complex patterns from data.",
        "Natural language processing enables text understanding.",
        "Large language models require extensive computational resources.",
    ] * 20
    
    # Find all checkpoint files
    checkpoint_paths = sorted(glob.glob(checkpoint_pattern))
    
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found matching pattern: {checkpoint_pattern}")
    
    print(f"Found {len(checkpoint_paths)} checkpoints")
    
    # Load one checkpoint to get vocab size and prepare data
    print("\nPreparing data...")
    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    model_args = first_checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    temp_model = GPT(gptconf)
    
    vocab_size = temp_model.transformer.wte.weight.shape[0]
    block_size = temp_model.config.block_size
    
    print(f"Model vocabulary: {vocab_size}")
    print(f"Model block size: {block_size}")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize
    max_length = min(128, block_size)
    input_ids_list = []
    
    for text in sample_texts:
        text = " ".join(text.split())
        tokens = enc.encode(text)
        tokens = tokens[:max_length]
        tokens = [min(max(t, 0), vocab_size - 1) for t in tokens]
        
        if len(tokens) > 0:
            input_ids_list.append(tokens)
    
    # Pad
    max_len = max(len(tokens) for tokens in input_ids_list)
    padded_input_ids = [tokens + [0] * (max_len - len(tokens)) 
                        for tokens in input_ids_list]
    
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    targets = input_ids.clone()
    
    print(f"Data shape: {input_ids.shape}")
    print(f"Token range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    
    # Clean up temporary model
    del temp_model, first_checkpoint
    torch.cuda.empty_cache()
    
    # Process each checkpoint
    results = []
    
    for checkpoint_path in checkpoint_paths:
        try:
            iter_num = extract_iter_num(checkpoint_path)
            min_sv, max_sv = compute_eigenvalues(checkpoint_path, input_ids, targets, device)
            
            results.append({
                'iter_num': iter_num,
                'min_sv': min_sv,
                'max_sv': max_sv,
                'condition_number': abs(max_sv/min_sv),
                'checkpoint': os.path.basename(checkpoint_path)
            })
            
        except Exception as e:
            print(f"Error processing {checkpoint_path}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('iter_num').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")
    print(df)
    
    return df


if __name__ == "__main__":
    # Example usage:
    # Checkpoint naming convention: <iter_num>_<other_info>.pt
    # e.g., 1000_model.pt, 2000_model.pt, 5000_model.pt
    
    checkpoint_pattern = "out-hessian-debug/*_*.pt"  # Adjust this pattern
    
    df = process_checkpoints(
        checkpoint_pattern=checkpoint_pattern,
        output_csv="hessian_eigenvalues.csv",
        device="cuda"
    )
    
    # Additional analysis
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(df[['iter_num', 'min_sv', 'max_sv']].describe())
