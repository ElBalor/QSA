"""Reference QSA language model: embed -> N x QSDecoderBlock -> Log-Born head.

Includes the full autoregressive generation loop backed by the complex KV
cache (one cache per layer). A ``kind='softmax'`` variant swaps every block
for a softmax equivalent (same body, standard LN + Linear head) so the
paper's controlled comparisons run end-to-end from one class.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .layers import QSDecoderBlock
from .baseline import SoftmaxDecoderBlock
from .heads import LogBornHead
from .cache import ComplexKVCache


@dataclass
class QSAConfig:
    vocab_size: int = 256          # byte-level default: no tokenizer needed
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = None               # defaults to 4 * d_model
    dropout: float = 0.05
    rope_base: float = 10000.0
    log_born_clamp: float = 50.0
    kind: str = "qsa"              # "qsa" | "softmax"


class QSAReferenceLM(nn.Module):
    """Decoder-only LM: token embedding -> N blocks -> Log-Born head.

    The imaginary stream is a learned per-token embedding (trainable), as in
    the paper's graft design where the imaginary stream is a trainable
    companion to the real hidden state. For ``kind='softmax'`` the imaginary
    stream is an inert zero tensor and the head is a standard Linear.
    """

    def __init__(self, config: QSAConfig = None, **overrides):
        super().__init__()
        config = config or QSAConfig()
        for k, v in overrides.items():
            if not hasattr(config, k):
                raise TypeError(f"Unknown config field: {k}")
            setattr(config, k, v)
        self.config = config
        d_ff = config.d_ff or 4 * config.d_model
        self.is_qsa = config.kind == "qsa"

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        if self.is_qsa:
            self.imag_emb = nn.Embedding(config.vocab_size, config.d_model)
            self.blocks = nn.ModuleList([
                QSDecoderBlock(config.d_model, config.n_heads, d_ff, config.dropout)
                for _ in range(config.n_layers)
            ])
            self.head = LogBornHead(config.d_model, config.vocab_size, clamp=config.log_born_clamp)
        else:
            self.imag_emb = None
            self.blocks = nn.ModuleList([
                SoftmaxDecoderBlock(config.d_model, config.n_heads, d_ff, config.dropout)
                for _ in range(config.n_layers)
            ])
            self.head = nn.Linear(config.d_model, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, attention_mask=None, targets=None):
        """idx: (B, T) int64 tokens. targets: (B, T) int64 next-token labels.

        Returns (logits, loss) where loss is None when targets is None.
        """
        B, T = idx.shape
        r = self.tok_emb(idx)
        if self.is_qsa:
            i = self.imag_emb(idx)
        else:
            i = torch.zeros_like(r)

        for block in self.blocks:
            r, i = block(r, i, attention_mask=attention_mask, rope_base=self.config.rope_base)

        if self.is_qsa:
            logits = self.head(r, i)
        else:
            logits = self.head(r)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """Autoregressive generation with the per-layer complex KV cache.

        The first forward pass processes the whole prompt (with cache
        creation); each subsequent step feeds a single token and appends to
        the caches — attending over the full history without recomputation.
        """
        self.eval()
        caches = None
        cur = idx
        for step in range(max_new_tokens):
            if caches is not None:
                cur = idx[:, -1:]  # only the newest token after the prompt pass
            r = self.tok_emb(cur)
            i = self.imag_emb(cur) if self.is_qsa else torch.zeros_like(r)

            new_caches = []
            for block, cache in zip(self.blocks, caches or [None] * len(self.blocks)):
                if cache is None:
                    out = block(r, i, use_cache=True, rope_base=self.config.rope_base)
                else:
                    out = block(r, i, past_kv=cache, use_cache=True, rope_base=self.config.rope_base)
                if len(out) == 3:
                    r, i, new_kv = out
                else:
                    r, i = out
                    new_kv = None
                new_caches.append(new_kv)
            caches = new_caches

            if self.is_qsa:
                logits = self.head(r, i)
            else:
                logits = self.head(r)
            logits = logits[:, -1, :]

            if temperature <= 0:  # greedy
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k, dim=-1)
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_tok], dim=1)
            if eos_id is not None and (next_tok == eos_id).all():
                break
        return idx
