import math
import torch
import torch.nn as nn

from .complex_linear import ComplexLinear
from .rope import apply_phase_rope


def cp_superposition(S_real, S_imag, v_r, v_i, d_k, dropout=None):
    """Core CP^{n-1} math: normalize complex scores and coherently superpose.

    S_*: (..., T_q, T_k) complex interference matrix (Hermitian product rows).
    v_*: (..., T_k, d_k) complex values.
    Returns (out_r, out_i): (..., T_q, d_k) complex superposition result.

    Step 3 (CP normalization): alpha = S / (||S_i||_2 * sqrt(d_k)) — the L2
    norm over each query's key row (the paper's Section 3.3 math) is fixed to
    1/sqrt(d_k); phase is fully preserved.
    Step 4 (coherent superposition): H = alpha @ V with complex multiplication,
    so negative/cancelling contributions are structural.

    Exposed as a pure function so the expressive-power properties are directly
    testable without fitting weight matrices.
    """
    S_norm = torch.sqrt((S_real**2 + S_imag**2).sum(dim=-1, keepdim=True) + 1e-8)
    alpha_real = S_real / (S_norm * math.sqrt(d_k))
    alpha_imag = S_imag / (S_norm * math.sqrt(d_k))

    if dropout is not None:
        alpha_real = dropout(alpha_real)
        alpha_imag = dropout(alpha_imag)

    # Coherent superposition: H = alpha * V (complex multiplication)
    out_r = torch.einsum("...qk,...kd->...qd", alpha_real, v_r) - torch.einsum(
        "...qk,...kd->...qd", alpha_imag, v_i
    )
    out_i = torch.einsum("...qk,...kd->...qd", alpha_real, v_i) + torch.einsum(
        "...qk,...kd->...qd", alpha_imag, v_r
    )
    return out_r, out_i


class QSAttention(nn.Module):
    """Quantum Self-Attention with complex projective normalization.

    Replaces softmax attention with:
      1. Complex embedding of Q, K, V (real input, zero imaginary stream).
      2. Hermitian interference matrix S = <Q, K>_C (replaces QK^T).
      3. CP^{n-1} normalization: alpha = S / (||S||_2 * sqrt(d_k)) — phase is
         preserved; no exponentials, no overflow.
      4. Coherent superposition H = sum_j alpha_ij * V~_j with native
         constructive/destructive interference.
      5. Born-rule readout: the real part of the output is returned downstream
         (the imaginary stream is available via ``return_imaginary``).

    Optional extras (all default-on/off flags keep the paper's bare path):
      * ``use_rope``: complex-native phase rotation of Q/K — the Hermitian
        product then depends only on relative position offsets.
      * ``past_kv``/``use_cache``: complex KV cache for incremental
        autoregressive generation.

    Future positions are masked with **zero**, not -inf: in CP^{n-1} space a
    zero amplitude contributes nothing to the norm and nothing to the output.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, causal=True, return_imaginary=False,
                 use_rope=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.return_imaginary = return_imaginary
        self.use_rope = use_rope

        self.wq = ComplexLinear(d_model, d_model)
        self.wk = ComplexLinear(d_model, d_model)
        self.wv = ComplexLinear(d_model, d_model)
        self.wo = ComplexLinear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.res_scale = 0.9  # residual scaling (QSA stability mechanism)

    def _split_heads(self, t):
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False, rope_base=10000.0):
        """x: (B, T_new, d_model) real tensor.

        attention_mask: optional (B, T_full) tensor with 1 for real tokens
        and 0 for padding (full forward only; ignored with a KV cache).

        past_kv: optional (k_r, k_i, v_r, v_i) from a previous step. When
        given, x holds only the new tokens and their keys/values are
        appended to the cache before attention over the full history.

        Returns:
            out (real) or (out_r, out_i) when ``return_imaginary``;
            if ``use_cache`` the result is wrapped as (out, new_kv).
        """
        B, T, C = x.shape
        x_imag = torch.zeros_like(x)

        # Project to complex Q, K, V
        q_r, q_i = self.wq(x, x_imag)
        k_r, k_i = self.wk(x, x_imag)
        v_r, v_i = self.wv(x, x_imag)

        q_r, q_i = self._split_heads(q_r), self._split_heads(q_i)
        k_r, k_i = self._split_heads(k_r), self._split_heads(k_i)
        v_r, v_i = self._split_heads(v_r), self._split_heads(v_i)

        cached = past_kv is not None

        if self.use_rope:
            if cached:
                q_pos = k_pos = past_kv.positions(T, device=x.device)  # S, S+1, ...
            else:
                q_pos = k_pos = torch.arange(T, device=x.device)
            q_r, q_i = apply_phase_rope(q_r, q_i, q_pos, base=rope_base)
            k_r, k_i = apply_phase_rope(k_r, k_i, k_pos, base=rope_base)

        if cached:
            past_kv.update(k_r.detach() if not torch.is_grad_enabled() else k_r,
                           k_i.detach() if not torch.is_grad_enabled() else k_i,
                           v_r.detach() if not torch.is_grad_enabled() else v_r,
                           v_i.detach() if not torch.is_grad_enabled() else v_i)
            k_r, k_i, v_r, v_i = past_kv.get()

        # Hermitian product: S = Q * K^H
        S_real = torch.einsum("bhqd,bhkd->bhqk", q_r, k_r) + torch.einsum("bhqd,bhkd->bhqk", q_i, k_i)
        S_imag = torch.einsum("bhqd,bhkd->bhqk", q_i, k_r) - torch.einsum("bhqd,bhkd->bhqk", q_r, k_i)

        # Causal mask (zero, not -inf — CP space masking). Only meaningful in
        # full-forward mode: with a KV cache every cached slot is in the past.
        if self.causal and not cached and k_r.shape[2] == T:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype)).view(1, 1, T, T)
            S_real = S_real * causal_mask
            S_imag = S_imag * causal_mask

        # Key padding mask: zero out amplitudes for padded keys
        if attention_mask is not None:
            key_mask = attention_mask.to(x.dtype).view(B, 1, 1, -1)
            S_real = S_real * key_mask
            S_imag = S_imag * key_mask

        # CP^{n-1} normalization + coherent superposition (dropout on amplitudes)
        out_r, out_i = cp_superposition(S_real, S_imag, v_r, v_i, self.d_k, dropout=self.dropout)

        # Merge heads
        out_r = out_r.transpose(1, 2).contiguous().view(B, T, C)
        out_i = out_i.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out_r, out_i = self.wo(out_r, out_i)

        out = (x + self.res_scale * out_r, self.res_scale * out_i) if self.return_imaginary \
            else x + self.res_scale * out_r

        if use_cache:
            from .cache import ComplexKVCache
            if cached:
                new_kv = past_kv
            else:
                new_kv = ComplexKVCache(B, self.n_heads, self.d_k, device=x.device, dtype=x.dtype)
                new_kv.update(k_r, k_i, v_r, v_i)
            return out, new_kv
        return out
