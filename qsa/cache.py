from __future__ import annotations

import torch


class ComplexKVCache:
    """Complex KV cache for efficient autoregressive QSA generation.

    Stores the phase-rotated complex keys and values per layer,

        (k_r, k_i, v_r, v_i)  each of shape (B, h, S, d_k),

    where S is the current sequence length, and appends new slots
    incrementally. This lets QSA attend to the full generation history
    without recomputing: exactly the "complex KV cache" of the paper's
    novelty list, generalized to the complex streams the mechanism needs.
    """

    def __init__(self, batch_size: int, n_heads: int, d_k: int, device=None, dtype=torch.float32):
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.d_k = d_k
        self.device = device
        self.dtype = dtype
        self.k_r = self.k_i = self.v_r = self.v_i = None
        self.length = 0

    def __len__(self) -> int:
        return self.length

    @torch.no_grad()
    def update(self, k_r, k_i, v_r, v_i) -> None:
        """Append new key/value slots: each (B, h, T_new, d_k)."""
        if self.k_r is None:
            self.k_r, self.k_i = k_r.clone(), k_i.clone()
            self.v_r, self.v_i = v_r.clone(), v_i.clone()
        else:
            self.k_r = torch.cat([self.k_r, k_r], dim=2)
            self.k_i = torch.cat([self.k_i, k_i], dim=2)
            self.v_r = torch.cat([self.v_r, v_r], dim=2)
            self.v_i = torch.cat([self.v_i, v_i], dim=2)
        self.length = self.k_r.shape[2]

    def get(self):
        """Current (k_r, k_i, v_r, v_i), each (B, h, S, d_k)."""
        return self.k_r, self.k_i, self.v_r, self.v_i

    def positions(self, new_len: int, device=None) -> torch.Tensor:
        """Absolute positions of the ``new_len`` incoming slots: S, S+1, ..."""
        return torch.arange(
            self.length, self.length + new_len, device=device or self.k_r.device
        )
