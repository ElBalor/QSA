import torch


def apply_phase_rope(r, i, positions, base=10000.0):
    """Complex-native rotary position embedding (phase rotation).

    Each per-head channel pair (c_{2j}, c_{2j+1}) is treated as a complex
    number and rotated by theta_{pos, j} = pos / base^(2j/d):

        c_j' = c_j * e^{i * theta_{pos, j}}

    This is exactly RoPE, derived natively in complex space: rotating Q by
    pos_q and K by pos_k makes the Hermitian product depend only on the
    *relative* offset (pos_q - pos_k), because e^{i theta_q} * conj(e^{i
    theta_k}) = e^{i (theta_q - theta_k)}.

    Args:
        r, i: (..., T, d_k) real/imag streams of Q or K (rotation happens in
            adjacent channel pairs, matching RoPE's pairing).
        positions: (T,) int64 absolute positions of each slot.
        base: wavelength base (same role as RoPE's theta base).

    Returns:
        (r_rot, i_rot) rotated streams, same shapes.
    """
    head_dim = r.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("phase RoPE requires an even per-head dimension")
    half = head_dim // 2

    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=r.device, dtype=torch.float32) * 2.0 / head_dim))
    theta = positions.to(device=r.device, dtype=torch.float32).unsqueeze(-1) * inv_freq  # (T, half)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Pair adjacent channels: (2j, 2j+1) as complex numbers
    r_even, r_odd = r[..., 0::2], r[..., 1::2]
    i_even, i_odd = i[..., 0::2], i[..., 1::2]

    # c = (r_even + i r_odd); multiply by (cos + i sin):
    #   new r_even = r_even cos - r_odd sin; new r_odd = r_even sin + r_odd cos
    #   imaginary stream rotates with the same rotation
    r_even_new = r_even * cos_t - r_odd * sin_t
    r_odd_new = r_even * sin_t + r_odd * cos_t
    i_even_new = i_even * cos_t - i_odd * sin_t
    i_odd_new = i_even * sin_t + i_odd * cos_t

    r_out = torch.stack([r_even_new, r_odd_new], dim=-1).flatten(-2)
    i_out = torch.stack([i_even_new, i_odd_new], dim=-1).flatten(-2)
    return r_out, i_out
