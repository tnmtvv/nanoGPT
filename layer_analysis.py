import torch
import tiktoken
from torch.autograd import grad
from model import GPT, GPTConfig  # nanoGPT imports
import scipy.sparse.linalg as spsplin
import pandas as pd
import glob
import re
import os
import argparse


def get_layer_parameters(model, layer_name):
    """Extract parameters for a specific layer."""
    params = []
    for name, param in model.named_parameters():
        if layer_name in name:
            params.append(param)
    return params


def hessian_vector_product_layer(model, input_ids, targets, vector, layer_params):
    """Compute Hessian-vector product for specific layer parameters."""
    model.zero_grad()
    
    vocab_size = model.transformer.wte.weight.shape[0]
    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    targets = torch.clamp(targets, 0, vocab_size - 1)
    
    _, loss = model(input_ids, targets)
    
    # Compute gradient only w.r.t. layer parameters
    grad = torch.autograd.grad(loss, layer_params, create_graph=True, allow_unused=True)
    grad = [g for g in grad if g is not None]
    
    if not grad:
        return None
    
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad])
    
    if vector.shape[0] != grad_vec.shape[0]:
        raise ValueError(f"Vector size {vector.shape[0]} doesn't match gradient size {grad_vec.shape[0]}")
    
    dot_product = torch.dot(grad_vec, vector)
    
    hvp = torch.autograd.grad(dot_product, layer_params, retain_graph=False)
    hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
    return hvp_flat



def _flatten(tensors):
    return torch.cat([t.contiguous().view(-1) for t in tensors])

@torch.no_grad()
def _count_params(params):
    return sum(p.numel() for p in params)

def _prepare_loss_and_first_grad(model, inputs, targets, layer_params):
    # model returns (logits, loss)
    _, loss = model(inputs, targets)
    grads = torch.autograd.grad(loss, layer_params, create_graph=True, retain_graph=True)
    g_flat = _flatten(grads)
    return loss, grads, g_flat

def _hvp_from_first_grad(grads, layer_params, v, retain_graph):
    # Given first grads, compute H*v via grad(<grads, v>)
    g_flat = _flatten(grads)
    gdotv = torch.dot(g_flat, v)
    hv = grad(gdotv, layer_params, retain_graph=retain_graph, create_graph=False)
    hv_flat = _flatten(hv)
    return hv_flat

def hutchinson_diag_and_offdiag_sum(
    model,
    inputs,
    targets,
    layer_params,      # iterable of parameters to differentiate w.r.t.
    m=100,             # number of Hutchinson probes
    use_rademacher=True
):
    """
    Returns:
      diag_est: (n,) tensor, Hutchinson diagonal estimate
      trace_est: float, estimated trace via same probes
      total_sum: float, 1^T H 1 over selected parameters
      offdiag_sum: float, (1^T H 1) - trace_est
    """
    device = next(model.parameters()).device
    n = _count_params(layer_params)

    # Compute first-order grads once and reuse for all HVPs
    loss, grads1, g1_flat = _prepare_loss_and_first_grad(model, inputs, targets, layer_params)

    diag_acc = torch.zeros(n, device=device, dtype=g1_flat.dtype)
    trace_acc = 0.0

    # Hutchinson probes for diag and trace
    for t in range(m):
        if use_rademacher:
            g = (torch.randint(0, 2, (n,), device=device, dtype=torch.int8) * 2 - 1).to(g1_flat.dtype)
        else:
            g = torch.randn(n, device=device, dtype=g1_flat.dtype)  # generalized estimator

        hv = _hvp_from_first_grad(grads1, layer_params, g, retain_graph=True)
        diag_acc += g * hv
        trace_acc += torch.dot(g, hv).item()

    diag_est = diag_acc / m
    trace_est = trace_acc / m  # equals sum(diag_est) in expectation

    # Compute total sum of all entries: 1^T H 1 (single HVP with ones vector)
    u = torch.ones(n, device=device, dtype=g1_flat.dtype)
    Hu = _hvp_from_first_grad(grads1, layer_params, u, retain_graph=False)
    total_sum = torch.dot(u, Hu).item()

    offdiag_sum = total_sum - trace_est
    ratio = abs(offdiag_sum / trace_est)

    return float(total_sum), float(ratio)



def compute_layer_eigenvalues(model, input_ids, targets, layer_params, device="cuda"):
    """Compute min and max eigenvalues for specific layer parameters. Lanczos algorithm."""
    num_params = sum(p.numel() for p in layer_params)
    
    if num_params == 0:
        return None, None, None
    
    print(f"Computing Hessian for {num_params:,} parameters...")
    
    def hvp_numpy(x):
        x_torch = torch.tensor(x, dtype=torch.float32).to(device)
        result = hessian_vector_product_layer(model, input_ids, targets, x_torch, layer_params)
        if result is None:
            return None
        return result.cpu().numpy()
    
    A = spsplin.LinearOperator((num_params, num_params), matvec=hvp_numpy)
    
    try:
        res_max = spsplin.eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-3, maxiter=100)
        max_sv = res_max[0]
        
        res_min = spsplin.eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-3, maxiter=100)
        min_sv = res_min[0]
        
        spectral_gap = abs(max_sv / min_sv) if abs(min_sv) > 1e-10 else float('inf')
        
        return min_sv, max_sv, spectral_gap
    
    except Exception as e:
        print(f"  Error computing eigenvalues: {e}")
        return None, None, None


def get_layer_names(model):
    """Extract unique layer identifiers from nanoGPT model."""
    layer_names = set()
    
    for name, _ in model.named_parameters():
        # nanoGPT structure: transformer.wte, transformer.wpe, transformer.h.X, transformer.ln_f
        if "transformer.h." in name:
            parts = name.split(".")
            # Get block identifier (e.g., "transformer.h.0")
        #     layer_id = ".".join(parts[:3])
        #     layer_names.add(layer_id)
        # elif "transformer.wte" in name:
        #     layer_names.add("transformer.wte")
        # elif "transformer.wpe" in name:
        #     layer_names.add("transformer.wpe")
        # elif "transformer.ln_f" in name:
        #     layer_names.add(name)
        # elif "lm_head" in name:
            layer_names.add(name)
    
    return sorted(list(layer_names))


def load_nanogpt_checkpoint(checkpoint_path, device="cuda"):
    """Load nanoGPT model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # nanoGPT checkpoint structure
    model_args = checkpoint['model_args']
    
    # Create GPTConfig using the dataclass constructor
    gptconf = GPTConfig(**model_args)
    
    # Disable flash attention if present (doesn't support double backward)
    if hasattr(gptconf, 'flash'):
        gptconf.flash = False
    
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    
    # Handle _orig_mod. prefix (from torch.compile)
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k.replace(unwanted_prefix, '')] = state_dict.pop(k)
    
    load_result = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)
    
    model.eval()
    model.to(device)
    
    return model, gptconf


def analyze_checkpoint_layers(checkpoint_path, input_ids, targets, device="cuda"):
    """Analyze all layers in a nanoGPT checkpoint."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")
    
    # Load nanoGPT model
    model, gptconf = load_nanogpt_checkpoint(checkpoint_path, device)
    
    input_ids_gpu = input_ids.to(device)
    targets_gpu = targets.to(device)
    
    # Get all layer names
    layer_names = get_layer_names(model)
    print(f"Found {len(layer_names)} layers to analyze")
    
    results = []
    
    # Force use of math backend for SDPA (supports double backward)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        # Test forward pass
        with torch.no_grad():
            logits, loss = model(input_ids_gpu, targets_gpu)
        print(f"✓ Forward pass successful! Loss: {loss.item():.4f}\n")
        
        # Analyze each layer
        for layer_name in layer_names:
            print(f"Analyzing layer: {layer_name}")
            
            layer_params = get_layer_parameters(model, layer_name)
            
            if not layer_params:
                print(f"  No parameters found for {layer_name}")
                continue
            
            # Compute diagonal/off-diagonal ratio
            print("  Computing diagonal dominance...")
            diag_sum, diag_ratio = hutchinson_diag_and_offdiag_sum(
                model, input_ids_gpu, targets_gpu, layer_params, m=100
            )
            
            if diag_ratio is not None:
                print(f"  Diagonal sum: {diag_sum:.6f}")
                print(f"  Diag/Off-diag ratio: {diag_ratio:.2f}")
            
            # Compute eigenvalues for spectral gap
            print("  Computing eigenvalues...")
            min_sv, max_sv, spectral_gap = compute_layer_eigenvalues(
                model, input_ids_gpu, targets_gpu, layer_params, device
            )
            
            if min_sv is not None:
                print(f"  Min eigenvalue: {min_sv:.6f}")
                print(f"  Max eigenvalue: {max_sv:.6f}")
                print(f"  Spectral gap: {spectral_gap:.2f}")
            
            # Check if layer is "bad"
            is_bad = False
            if diag_ratio is not None and diag_ratio >= 100:
                is_bad = True
                print(f"  ⚠️  BAD LAYER: High diagonal dominance ({diag_ratio:.2f})")
            if spectral_gap is not None and spectral_gap >= 100:
                is_bad = True
                print(f"  ⚠️  BAD LAYER: Large spectral gap ({spectral_gap:.2f})")
            
            results.append({
                'layer_name': layer_name,
                'num_params': sum(p.numel() for p in layer_params),
                'diag_sum': diag_sum,
                'diag_off_diag_ratio': diag_ratio,
                'min_eigenvalue': min_sv,
                'max_eigenvalue': max_sv,
                'spectral_gap': spectral_gap,
                'is_bad_layer': is_bad
            })
            
            print()
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def extract_iter_num(checkpoint_path):
    """Extract iteration number from checkpoint filename."""
    filename = os.path.basename(checkpoint_path)
    match = re.search(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        # Try alternative pattern: ckpt_<iter>.pt or iter_<iter>.pt
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract iteration number from {filename}")


# def process_checkpoints_layerwise(checkpoint_pattern, output_csv="hessian_layer_results.csv", device="cuda"):
#     """Process multiple nanoGPT checkpoints with layer-wise analysis."""
    
#     # Sample texts
#     sample_texts = [
#         "The quick brown fox jumps over the lazy dog.",
#         "Machine learning transforms artificial intelligence research.",
#         "Deep neural networks learn complex patterns from data.",
#         "Natural language processing enables text understanding.",
#         "Large language models require extensive computational resources.",
#     ] * 20
def process_checkpoints_layerwise(checkpoint_pattern, output_csv="hessian_layer_results.csv", device="cuda"):
    """Process multiple nanoGPT checkpoints with layer-wise analysis."""

    # Coriolanus sample text (public domain)
    speech = """
    MARCIUS:
    He that will give good words to thee will flatter
    Beneath abhorring. What would you have, you curs,
    That like nor peace nor war? the one affrights you,
    The other makes you proud. He that trusts to you,
    Where he should find you lions, finds you hares;
    Where foxes, geese: you are no surer, no,
    Than is the coal of fire upon the ice,
    Or hailstone in the sun. Your virtue is
    To make him worthy whose offence subdues him
    And curse that justice did it.
    Who deserves greatness
    Deserves your hate; and your affections are
    A sick man's appetite, who desires most that
    Which would increase his evil. He that depends
    Upon your favours swims with fins of lead
    And hews down oaks with rushes. Hang ye! Trust Ye?
    With every minute you do change a mind,
    And call him noble that was now your hate,
    Him vile that was your garland. What's the matter,
    That in these several places of the city
    You cry against the noble senate, who,
    Under the gods, keep you in awe, which else
    Would feed on one another? What's their seeking?

    MENENIUS:
    For corn at their own rates; whereof, they say,
    The city is well stored.

    MARCIUS:
    Hang 'em! They say!
    They'll sit by the fire, and presume to know
    What's done i' the Capitol; who's like to rise,
    Who thrives and who declines; side factions and give out
    Conjectural marriages; making parties strong
    And feebling such as stand not in their liking
    Below their cobbled shoes. They say there's grain enough!
    Would the nobility lay aside their ruth,
    And let me use my sword, I'll make a quarry
    With thousands of these quarter'd slaves, as high
    As I could pick my lance.

    MENENIUS:
    Nay, these are almost thoroughly persuaded;
    For though abundantly they lack discretion,
    Yet are they passing cowardly. But, I beseech you,
    What says the other troop?
    """.strip()

    # Split into non-empty lines; optionally repeat to get more samples
    sample_texts = [line.strip() for line in speech.split("\n") if line.strip()]
    
    checkpoint_paths = sorted(glob.glob(checkpoint_pattern))
    
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found matching pattern: {checkpoint_pattern}")
    
    print(f"Found {len(checkpoint_paths)} checkpoints")
    
    # Prepare data using first checkpoint
    print("\nPreparing data...")
    model, gptconf = load_nanogpt_checkpoint(checkpoint_paths[0], "cpu")
    
    vocab_size = gptconf.vocab_size
    block_size = gptconf.block_size
    
    print(f"Model vocabulary: {vocab_size}")
    print(f"Model block size: {block_size}")
    
    enc = tiktoken.get_encoding("gpt2")
    
    max_length = min(128, block_size)
    input_ids_list = []
    
    for text in sample_texts:
        text = " ".join(text.split())
        tokens = enc.encode(text)
        tokens = tokens[:max_length]
        tokens = [min(max(t, 0), vocab_size - 1) for t in tokens]
        
        if len(tokens) > 0:
            input_ids_list.append(tokens)
    
    max_len = max(len(tokens) for tokens in input_ids_list)
    padded_input_ids = [tokens + [0] * (max_len - len(tokens)) 
                        for tokens in input_ids_list]
    
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    targets = input_ids.clone()
    
    print(f"Data shape: {input_ids.shape}")
    print(f"Token range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    
    del model
    torch.cuda.empty_cache()
    
    # Process each checkpoint
    all_results = []
    
    for checkpoint_path in checkpoint_paths:
        try:
            iter_num = extract_iter_num(checkpoint_path)
            layer_results = analyze_checkpoint_layers(checkpoint_path, input_ids, targets, device)
            
            for result in layer_results:
                result['iter_num'] = iter_num
                result['checkpoint'] = os.path.basename(checkpoint_path)
                all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    df = df.sort_values(['iter_num', 'layer_name']).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")
    print(df)
    
    # Summary of bad layers
    if 'is_bad_layer' in df.columns:
        bad_layers = df[df['is_bad_layer'] == True]
        print(f"\n{'='*60}")
        print(f"Found {len(bad_layers)} bad layer instances:")
        print(f"{'='*60}")
        if len(bad_layers) > 0:
            print(bad_layers[['iter_num', 'layer_name', 'diag_off_diag_ratio', 'spectral_gap']])
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="out-warm_up",
        help="Folder containing *_*.pt checkpoints",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="hessian_layer_analysis_warm_up.csv",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    args = parser.parse_args()  # parses CLI args 

    checkpoint_pattern = os.path.join(args.checkpoint_dir, "*_*.pt")

    df = process_checkpoints_layerwise(
        checkpoint_pattern=checkpoint_pattern,
        output_csv=args.output_csv,
        device=args.device,
    )

    print("\n" + "="*60)
    print("Summary Statistics by Layer:")
    print("="*60)
    print(df.groupby('layer_name')[['diag_off_diag_ratio', 'spectral_gap']].describe())


if __name__ == "__main__":
    main()
