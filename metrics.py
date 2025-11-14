import math 
import torch 


@torch.no_grad()
def compute_perplexity(loss):
    """
    Compute perplexity from cross-entropy loss.
    Perplexity = exp(loss)
    Lower perplexity indicates better model performance.
    """
    return torch.exp(loss).item()


@torch.no_grad()
def compute_bits_per_character(loss, tokens_per_char=1.0):
    """
    Compute bits-per-character (BPC) metric.
    BPC = loss / log(2) / tokens_per_char
    This normalizes the loss to bits and adjusts for character tokenization.
    """
    return (loss.item() / math.log(2)) / tokens_per_char


@torch.no_grad()
def compute_token_accuracy(logits, targets):
    """
    Compute token-level accuracy (percentage of correctly predicted tokens).
    """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    accuracy = (correct / total).item()
    return accuracy * 100  # Return as percentage


@torch.no_grad()
def compute_top_k_accuracy(logits, targets, k=5):
    """
    Compute top-k accuracy (whether correct token is in top-k predictions).
    """
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=-1).float().sum()
    total = targets.numel()
    return (correct / total).item() * 100


@torch.no_grad()
def compute_confidence_metrics(logits):
    """
    Compute confidence-related metrics:
    - Mean confidence: average probability assigned to predicted tokens
    - Entropy: average entropy of probability distribution
    """
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1)
    mean_confidence = max_probs.mean().item()
    
    # Compute entropy: -sum(p * log(p))
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
    
    return mean_confidence, entropy

