import torch
import torch.nn as nn

from .complex_linear import ComplexLinear


class LogBornHead(nn.Module):
    """Log-Born readout (replaces the standard LM projection).

    Maps a complex hidden state (r, i) into real logits:

        logits = W_head @ log(r^2 + i^2 + eps)

    The logarithm cancels the exponential in cross-entropy, yielding stable
    gradients via Wirtinger calculus. Output is clamped to [-50, 50] as the
    paper's final safety net.
    """

    def __init__(self, d_model, vocab_size, clamp=50.0, eps=1e-5):
        super().__init__()
        self.head = nn.Linear(d_model, vocab_size)
        self.clamp = clamp
        self.eps = eps

    def forward(self, r, i):
        magnitude_sq = r * r + i * i
        log_born = torch.log(magnitude_sq + self.eps)
        logits = self.head(log_born)
        return torch.clamp(logits, -self.clamp, self.clamp)


class ComplexClassifierHead(nn.Module):
    """Born-rule readout for sequence classification.

    Takes the real stream (optionally pooled) and projects to class logits.
    Use when the downstream task is classification rather than LM.
    """

    def __init__(self, d_model, num_classes, pool="mean"):
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, r, i, attention_mask=None):
        if self.pool == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(r.dtype)
                pooled = (r * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                pooled = r.mean(dim=1)
        else:  # last non-pad token via 'last'
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).long().clamp(min=1)
                pooled = r[torch.arange(r.size(0), device=r.device), lengths - 1]
            else:
                pooled = r[:, -1]
        return self.proj(pooled)
