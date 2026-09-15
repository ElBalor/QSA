"""Harmonic interference benchmark from the QSA paper.

Task: classify two superimposed sine waves by their interference type —
constructive (relative phase ~ 0) vs destructive (relative phase ~ pi).

Both baselines get the identical capacity (d_model=32, 4 heads, 1 attention
layer, same input embedding and head) so the comparison isolates the
attention mechanism. Phase information is presented explicitly through
sin/cos channels of each component wave; the input embedding is trainable
and shared.

The provable separation claim (softmax cannot cancel; QSA can, exactly) is
covered analytically in tests/test_qsa.py::test_qsa_is_subtractive.

Usage:
    python benchmarks/harmonic_interference.py [--epochs 40] [--n 4000]
"""

import argparse
import math
import torch
import torch.nn as nn

from qsa import QSAttention, spectral_clamp


def make_batch(n, T=64, seed=None):
    """Phase-tagged two-wave superposition.

    Channels per timestep: [sin(f1 t), cos(f1 t), sin(f2 t + dphi), cos(f2 t + dphi)]
    Label y=1: constructive (dphi ~ 0); y=0: destructive (dphi ~ pi).
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0)          # (1, T)
    f1 = (torch.rand(n, generator=g).unsqueeze(1) * 2.0 + 0.5)  # (n, 1)
    f2 = (torch.rand(n, generator=g).unsqueeze(1) * 2.0 + 0.5)
    y = torch.randint(0, 2, (n,), generator=g)
    dphi = torch.where(
        y.bool(),
        (torch.rand(n, generator=g) - 0.5) * 0.2,                     # near-aligned
        math.pi + (torch.rand(n, generator=g) - 0.5) * 0.2,           # near-opposed
    ).unsqueeze(1)
    p1 = f1 * t
    p2 = f2 * t + dphi
    x = torch.stack(
        [torch.sin(p1), torch.cos(p1), torch.sin(p2), torch.cos(p2)], dim=-1
    )  # (n, T, 4)
    return x, y


class AttentionClassifier(nn.Module):
    """Shared body: input embedding -> 1 attention layer -> pooled MLP head.

    ``attention_kind`` selects the mechanism under test; everything else is
    identical for a controlled comparison.
    """

    def __init__(self, attention_kind, d_model=32, n_heads=4):
        super().__init__()
        self.embed = nn.Linear(4, d_model)
        if attention_kind == "qsa":
            self.attn = QSAttention(d_model=d_model, n_heads=n_heads, causal=False)
        else:
            self.attn = SoftmaxAttention(d_model=d_model, n_heads=n_heads)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def forward(self, x):
        h = self.embed(x)
        h = self.attn(h)
        return self.head(h.mean(dim=1))


class SoftmaxAttention(nn.Module):
    """Plain real-valued attention with softmax — the paper's baseline."""

    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.h = n_heads
        self.dk = d_model // n_heads

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        a = torch.softmax(scores, dim=-1)
        out = (a @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o(out)


def train(model, loader, epochs, use_spectral_clamp=False, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            if use_spectral_clamp:
                spectral_clamp(model, gamma=1.0)


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        correct += (model(xb).argmax(dim=1) == yb).sum().item()
        total += yb.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x_train, y_train = make_batch(args.n, seed=1)
    x_test, y_test = make_batch(args.n // 4, seed=2)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), batch_size=512,
    )

    print(f"n_train={args.n} epochs={args.epochs} d_model=32 heads=4 (shared body)\n")

    results = {}
    for name in ["softmax", "qsa"]:
        torch.manual_seed(args.seed)
        model = AttentionClassifier(name)
        train(model, train_loader, args.epochs, use_spectral_clamp=(name == "qsa"))
        acc = accuracy(model, test_loader)
        results[name] = acc
        print(f"{name:>8}: test_acc={acc*100:6.2f}%")

    gap = results["qsa"] - results["softmax"]
    print(f"\nQSA advantage: {gap*100:+.2f} percentage points")


if __name__ == "__main__":
    main()
