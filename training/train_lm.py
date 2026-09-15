"""QSA language-model training script — the "accompanying repository" code
from the paper, made runnable.

Default task: **causal arithmetic** (the paper's stability stress-test).
A byte-level decoder-only LM learns arithmetic expressions like
``12+7-3=16<nl>``. Train on short expressions (few operators), watch
generalization on longer ones. Includes the paper's stability regimen:
spectral clamping after every optimizer step, warmup -> cosine LR, and
gradient clipping.

Examples:
    # train both mechanisms and compare (small, CPU-friendly)
    python training/train_lm.py --n_layers 4 --d_model 128 --steps 2000

    # larger single run
    python training/train_lm.py --kind qsa --steps 20000 --device cuda
"""

import argparse
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsa import QSAReferenceLM, QSAConfig, spectral_clamp

EOS_BYTE = 10  # newline terminates each expression


# ---------------------------------------------------------------------------
# Data: causal arithmetic, byte-level
# ---------------------------------------------------------------------------

def gen_expression(n_terms, lo=1, hi=99, rng=None):
    rng = rng or random
    parts = [str(rng.randint(lo, hi))]
    ops = ["+", "-"]
    for _ in range(n_terms - 1):
        parts.append(rng.choice(ops))
        parts.append(str(rng.randint(lo, hi)))
    return "".join(parts)


def eval_expression(expr):
    # Expressions are built as a - b + c ...; Python eval is fine on these
    return eval(expr)


def make_dataset(n_samples, min_terms=2, max_terms=5, seed=0):
    """Returns (text, byte_tensor). Each line: 'expr=answer' + newline."""
    rng = random.Random(seed)
    lines = []
    for _ in range(n_samples):
        n_terms = rng.randint(min_terms, max_terms)
        expr = gen_expression(n_terms, rng=rng)
        ans = eval_expression(expr)
        lines.append(f"{expr}={ans}\n")
    text = "".join(lines)
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    return text, data


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def ood_batch(n_samples, min_terms, max_terms, seed):
    """Held-out longer expressions for generalization eval."""
    return make_dataset(n_samples, min_terms=min_terms, max_terms=max_terms, seed=seed)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(model, data, block_size, device, max_batches=20):
    model.eval()
    losses = []
    for _ in range(max_batches):
        x, y = get_batch(data, block_size, 64, device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"[qsa-train] device={device} kind={args.kind}")
    _, train_data = make_dataset(args.n_train, min_terms=2, max_terms=args.max_terms, seed=1)
    _, ood_data = ood_batch(2000, min_terms=args.max_terms + 1, max_terms=args.max_terms + 3, seed=42)

    config = QSAConfig(
        vocab_size=256,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        kind=args.kind,
    )
    model = QSAReferenceLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[qsa-train] params={n_params/1e6:.2f}M block={args.block_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(10, int(0.05 * args.steps))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    model.train()
    nan_hits = 0
    for step in range(args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, targets=y)
        if not torch.isfinite(loss):
            nan_hits += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        spectral_clamp(model, gamma=1.0)  # paper stability regimen

        if step % args.log_every == 0 or step == args.steps - 1:
            train_loss = loss.item()
            ood_loss = evaluate_loss(model, ood_data, args.block_size, device, max_batches=5)
            print(
                f"step {step:6d} | lr {lr:.2e} | train {train_loss:7.4f} "
                f"| ood({args.max_terms+1}-{args.max_terms+3} ops) {ood_loss:7.4f} | NaNs {nan_hits}"
            )

    # Sample a few completions
    model.eval()
    prompt = "12+7-3="
    ctx = torch.tensor(list(prompt.encode()), dtype=torch.long).unsqueeze(0).to(device)
    out = model.generate(ctx, max_new_tokens=12, temperature=0.0, eos_id=EOS_BYTE)
    print(f"[sample] {prompt!r} -> {out[0].tolist()!r}")
    print(f"[sample decoded] {bytes(out[0].tolist()).decode('utf-8', errors='replace')!r}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{args.kind}_lm.pt")
        torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, path)
        print(f"[qsa-train] saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["qsa", "softmax", "both"], default="both")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--max_terms", type=int, default=5)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    kinds = ["qsa", "softmax"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        print(f"\n=== training {kind} ===")
        args.kind = kind
        train(args)


if __name__ == "__main__":
    main()
