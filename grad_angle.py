# gradient_angle_layers.py  (FULL-GRADIENT angle across all params)
import os, re, glob
import torch
import torch.nn.functional as F
import pandas as pd

def iter_num(p):
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1

def angle_cos(v1, v2, eps=1e-12):
    dot = torch.dot(v1, v2)
    cos = dot / (torch.norm(v1) * torch.norm(v2) + eps)
    return float(cos.item())

def model_ctor_from_nanogpt_ckpt(ckpt):
    from model import GPT, GPTConfig
    conf = GPTConfig(**ckpt["model_args"])
    if hasattr(conf, "flash"):
        conf.flash = False
    return GPT(conf)


def make_batch_from_text(text, vocab_size, block_size, device, enc):
    ids = enc.encode(text)
    ids = [min(max(t, 0), vocab_size - 1) for t in ids]
    ids = ids[: block_size + 1]  # +1 for shift
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device)[None, :]  # (1, T)
    y = torch.tensor(ids[1:],  dtype=torch.long, device=device)[None, :]  # (1, T)
    return x, y

def load_full_grad_vec(ckpt_path, model_ctor, x, y, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # [web:169]
    model = model_ctor(ckpt).to(device)

    sd = ckpt["model"] if "model" in ckpt else ckpt
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()

    # IMPORTANT: use named_parameters() so ordering is stable across runs
    g = torch.cat([
        p.grad.detach().flatten()
        for _, p in model.named_parameters()
        if p.grad is not None
    ])
    return g

def collect_grad_vec(model, name_prefix: str):
    # Concatenate grads for parameters whose names start with name_prefix
    parts = []
    for name, p in model.named_parameters():  # stable ordering [web:14]
        if not name.startswith(name_prefix):
            continue
        if p.grad is None:
            continue
        parts.append(p.grad.detach().flatten())
    if not parts:
        return None
    return torch.cat(parts)

def cos_sim(v1, v2, eps=1e-12):
    dot = torch.dot(v1, v2)
    denom = torch.norm(v1) * torch.norm(v2) + eps
    return float((dot / denom).item())

def load_attn_block_grad_vecs(ckpt_path, model_ctor, x, y, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = model_ctor(ckpt).to(device)

    sd = ckpt["model"] if "model" in ckpt else ckpt
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()

    # infer number of layers from config (nanoGPT-style) [web:6]
    n_layer = model.config.n_layer

    out = {}
    for i in range(n_layer):
        prefix = f"transformer.h.{i}.attn."  # nanoGPT param naming [web:6]
        g = collect_grad_vec(model, prefix)
        out[i] = g  # may be None if something is off / frozen
    return out


# ---------------- configure ----------------
dev = "cuda"
out_csv = "block_grad_angle_adagram_adamw.csv"

dir_B = "out-minigpt"
dir_A = "out-hessian-adagram-best"
# dir_B = "out-sgd"
# -------------------------------------------

ckpts_A = sorted(glob.glob(os.path.join(dir_A, "*_ckpt.pt")), key=iter_num)
ckpts_B = sorted(glob.glob(os.path.join(dir_B, "*_ckpt.pt")), key=iter_num)

# build ONE fixed batch for all checkpoints
import tiktoken
enc = tiktoken.get_encoding("gpt2")

ckpt0 = torch.load(ckpts_A[0], map_location="cpu", weights_only=False)  # [web:169]
vocab_size = ckpt0["model_args"]["vocab_size"]
block_size = ckpt0["model_args"]["block_size"]
x, y = make_batch_from_text(
    text="""MARCIUS:
        He that will give good words to thee will flatter
        Beneath abhorring. What would you have, you curs,
        That like nor peace nor war? the one affrights you,
        The other makes you proud. He that trusts to you,
        Where he should find you lions, finds you hares;""",
    vocab_size=vocab_size,
    block_size=min(128, block_size),
    device=dev,
    enc=enc,
)

rows = []
for pa, pb in zip(ckpts_A, ckpts_B):
    step_a, step_b = iter_num(pa), iter_num(pb)
    if step_a != step_b:
        continue

#     ga = load_full_grad_vec(pa, model_ctor_from_nanogpt_ckpt, x, y, device=dev)
#     gb = load_full_grad_vec(pb, model_ctor_from_nanogpt_ckpt, x, y, device=dev)

#     rows.append({
#         "iter": step_a,
#         "grad_angle_cos": angle_cos(ga, gb),
#         "ckpt_A": os.path.basename(pa),
#         "ckpt_B": os.path.basename(pb),
#         "grad_dim": int(ga.numel()),
#     })
    ga_blocks = load_attn_block_grad_vecs(pa, model_ctor_from_nanogpt_ckpt, x, y, device=dev)
    gb_blocks = load_attn_block_grad_vecs(pb, model_ctor_from_nanogpt_ckpt, x, y, device=dev)

    for i in ga_blocks.keys():
        g1, g2 = ga_blocks[i], gb_blocks[i]
        if g1 is None or g2 is None or g1.numel() != g2.numel():
            continue
        rows.append({
            "iter": step_a,
            "block": i,
            "attn_grad_angle_cos": cos_sim(g1, g2),
            "grad_dim": int(g1.numel()),
            "ckpt_A": os.path.basename(pa),
            "ckpt_B": os.path.basename(pb),
        })


pd.DataFrame(rows).sort_values("iter").to_csv(out_csv, index=False)  # [web:172]
print("saved:", out_csv)

