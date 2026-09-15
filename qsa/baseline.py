"""Reference implementations of standard softmax attention.

Provided so the paper's controlled comparisons (same body, same head, only
the attention mechanism swapped) can be run end-to-end from this package.
"""

import math
import torch
import torch.nn as nn


class SoftmaxAttention(nn.Module):
    """Softmax attention returning (real, imag) streams.

    The imaginary stream is returned unchanged (zero) so it can serve as a
    drop-in swap for QSAttention inside QSDecoderBlock — the paper's
    "same body, only the mechanism differs" comparison.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, causal=True, use_rope=True, **_):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.use_rope = use_rope

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.res_scale = 0.9

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False, rope_base=10000.0):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Real-only RoPE (rotate pairs) so positional handling matches QSA
        if self.use_rope:
            from .rope import apply_phase_rope
            zero_i = torch.zeros_like(q)
            if past_kv is not None:
                q_pos = k_pos = past_kv.positions(T, device=x.device)
            else:
                q_pos = k_pos = torch.arange(T, device=x.device)
            q_r, _ = apply_phase_rope(q, zero_i, q_pos, base=rope_base)
            k_r, _ = apply_phase_rope(k, zero_i, k_pos, base=rope_base)
            q, k = q_r, k_r

        if past_kv is not None:
            past_kv.update(k, torch.zeros_like(k), v, torch.zeros_like(v))
            k, _, v, _ = past_kv.get()

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if self.causal and k.shape[2] == T:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype)).view(1, 1, T, T)
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        if attention_mask is not None:
            key_mask = attention_mask.to(x.dtype).bool().view(B, 1, 1, -1)
            scores = scores.masked_fill(~key_mask, float("-inf"))
        a = torch.softmax(scores, dim=-1)
        a = self.dropout(a)
        out = (a @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.o(out)

        result = x + self.res_scale * out
        if use_cache:
            if past_kv is None:
                from .cache import ComplexKVCache
                past_kv = ComplexKVCache(B, self.n_heads, self.d_k, device=x.device, dtype=x.dtype)
                past_kv.update(k, torch.zeros_like(k), v, torch.zeros_like(v))
            return result, past_kv
        return result


class SoftmaxDecoderBlock(nn.Module):
    """Pre-norm softmax block with the same wiring as QSDecoderBlock.

    Tracks a zero imaginary stream for interface parity with the QSA path.
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.05):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SoftmaxAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.res_scale = 0.9

    def forward(self, r, i, attention_mask=None, past_kv=None, use_cache=False, rope_base=10000.0):
        h = self.ln1(r)
        attn_out = self.attn(
            h, attention_mask=attention_mask, past_kv=past_kv,
            use_cache=use_cache, rope_base=rope_base,
        )
        if use_cache:
            attn_r, new_kv = attn_out
        else:
            attn_r, new_kv = attn_out, None
        r = r + self.res_scale * attn_r

        ff = self.ffn(self.ln2(r))
        r = r + self.res_scale * ff
        return (r, i, new_kv) if use_cache else (r, i)
