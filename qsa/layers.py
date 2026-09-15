import torch
import torch.nn as nn

from .attention import QSAttention
from .complex_linear import ComplexLinear


class ComplexLayerNorm(nn.Module):
    """LayerNorm for complex-valued tensors (real + imaginary streams).

    Normalizes each stream independently over the feature dimension with
    separate learnable affine parameters. Used to prevent phase collapse in
    deep QSA stacks (paper: "Phase collapse is prevented by
    ComplexLayerNorm, stable initialisation, and optional reconstruction
    loss").
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.ln_r = nn.LayerNorm(d_model, eps=eps)
        self.ln_i = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x_real, x_imag):
        return self.ln_r(x_real), self.ln_i(x_imag)


class ComplexFeedForward(nn.Module):
    """Position-wise feed-forward network on complex streams.

    Real stream passes through GELU; the imaginary stream rides the same
    activations' sign structure via a parallel path, then both are mixed by a
    complex output projection. Residual scaling of 0.9 per the paper.
    """

    def __init__(self, d_model, d_ff=None, dropout=0.05):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.up = ComplexLinear(d_model, d_ff)
        self.down = ComplexLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.res_scale = 0.9

    def forward(self, x_real, x_imag):
        h_r, h_i = self.up(x_real, x_imag)
        h_r = torch.nn.functional.gelu(h_r)
        h_i = torch.nn.functional.gelu(h_i)
        h_r = self.dropout(h_r)
        h_i = self.dropout(h_i)
        out_r, out_i = self.down(h_r, h_i)
        return x_real + self.res_scale * out_r, x_imag + self.res_scale * out_i


class QSDecoderBlock(nn.Module):
    """QSA decoder block: ComplexLayerNorm -> QSAttention -> residual add ->
    ComplexLayerNorm -> ComplexFeedForward -> residual add.

    Accepts a real hidden state ``r`` and a (possibly trainable) imaginary
    stream ``i`` — exactly the interface used in the X graft onto frozen
    DeepSeek-Coder layers — and returns the complex update (dr, di).
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.05):
        super().__init__()
        self.ln1 = ComplexLayerNorm(d_model)
        self.attn = QSAttention(d_model, n_heads, dropout=dropout, return_imaginary=True)
        self.ln2 = ComplexLayerNorm(d_model)
        self.ffn = ComplexFeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, r, i, attention_mask=None, past_kv=None, use_cache=False, rope_base=10000.0):
        # Pre-norm attention over the real stream. QSAttention adds the real
        # residual internally; the imaginary stream gets its residual here.
        r_norm, _ = self.ln1(r, i)
        attn_out = self.attn(
            r_norm,
            attention_mask=attention_mask,
            past_kv=past_kv,
            use_cache=use_cache,
            rope_base=rope_base,
        )
        if use_cache:
            (attn_r, attn_i), new_kv = attn_out
        else:
            attn_r, attn_i = attn_out
            new_kv = None
        r = attn_r
        i = i + attn_i

        # Pre-norm feed-forward on both streams
        r_norm, i_norm = self.ln2(r, i)
        ff_r, ff_i = self.ffn(r_norm, i_norm)
        r = ff_r
        i = ff_i
        return (r, i, new_kv) if use_cache else (r, i)
