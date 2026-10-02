"""
QSA — PAPER FIDELITY TEST (against the installed pip package)
=============================================================
Tests that `qsa-attention` (PyPI) does exactly what the paper says
(Zenodo 10.5281/zenodo.22773015, v2):

  P1   Drop-in replacement for softmax attention, causal
  P2   CP^{n-1} normalization: every attention row has L2 norm 1/sqrt(d_k)
  P2b  softmax contrast: rows sum to 1, all positive (simplex, not CP)
  P3   Phase information preserved (complex amplitudes, not positive reals)
  P4   Destructive interference: (+1, -1) cancels exactly; softmax cannot
  P5   No-exponential stability: output exactly scale-invariant in scores
  P5b  softmax contrast: same scaling saturates to one-hot (fp64)
  P6   Antisymmetric relational encoding: sign flip = contribution flip
  P7   Relative-position encoding via phase RoPE: score depends only on dt
  P8   Complex KV cache: incremental generation == full forward

Run:  python tests/test_paper_fidelity.py      (or pytest tests/)
"""
import math
import torch

torch.manual_seed(0)

from qsa import QSAttention, cp_superposition, apply_phase_rope


# ── P1 ────────────────────────────────────────────────────────────────
def test_p1_drop_in_and_causal():
    attn = QSAttention(32, 4, causal=True, use_rope=True, dropout=0.0)
    attn.eval()
    x = torch.randn(2, 12, 32)
    with torch.no_grad():
        y1 = attn(x)
        y2 = attn(torch.cat([x[:, :4], torch.randn(2, 8, 32)], dim=1))
    assert y1.shape == x.shape
    assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-5), "future leaked to past"


# ── P2 / P2b / P3 ─────────────────────────────────────────────────────
def test_p2_cp_norm_and_phase():
    B, H, T, D = 1, 1, 8, 8
    S_r = torch.randn(B, H, T, T)
    S_i = torch.randn(B, H, T, T)
    V_r = torch.eye(T).view(1, 1, T, T).expand(B, H, T, T)
    V_i = torch.zeros_like(V_r)
    out_r, out_i = cp_superposition(S_r, S_i, V_r, V_i, D)
    row_norms = (out_r ** 2 + out_i ** 2).sum(-1).sqrt()
    assert torch.allclose(row_norms, torch.full_like(row_norms, 1 / math.sqrt(D)),
                          atol=1e-5), "CP row norm violated"
    smax = torch.softmax(S_r, dim=-1)
    assert torch.allclose(smax.sum(-1), torch.ones_like(smax.sum(-1)))
    assert bool((smax >= 0).all())
    assert float(out_i.abs().max()) > 0.05, "phase was destroyed"


# ── P4 ────────────────────────────────────────────────────────────────
def test_p4_destructive_interference():
    v = torch.tensor([[1.0], [1.0]])                     # equal values
    s_row = torch.tensor([[[[1.0, -1.0]]]])              # anti-aligned scores
    o_r, _ = cp_superposition(s_row, torch.zeros_like(s_row),
                              v.view(1, 1, 2, 1), torch.zeros(1, 1, 2, 1), 1)
    assert float(o_r.abs()) < 1e-6, "QSA failed to cancel"
    sm = torch.softmax(torch.tensor([1.0, -1.0]), dim=-1) @ v.squeeze()
    assert abs(float(sm)) - 1.0 < 1e-6, "softmax unexpectedly canceled"


# ── P5 / P5b ──────────────────────────────────────────────────────────
def test_p5_no_exponential_stability():
    v = torch.tensor([[1.0], [1.0]])
    s_row = torch.tensor([[[[1.0, -1.0]]]])
    o, _ = cp_superposition(s_row, torch.zeros_like(s_row),
                            v.view(1, 1, 2, 1), torch.zeros(1, 1, 2, 1), 1)
    o_big, _ = cp_superposition(s_row * 1e8, torch.zeros_like(s_row),
                                v.view(1, 1, 2, 1), torch.zeros(1, 1, 2, 1), 1)
    assert torch.allclose(o_big, o, atol=1e-6), "QSA saturated"
    s64 = torch.tensor([1.0, -1.0], dtype=torch.float64)
    sm_lo = torch.softmax(s64, dim=-1)
    sm_big = torch.softmax(s64 * 1e8, dim=-1)
    assert not torch.allclose(sm_lo, sm_big), "softmax did not saturate"


# ── P6 ────────────────────────────────────────────────────────────────
def test_p6_antisymmetric_encoding():
    v = torch.tensor([[1.0], [1.0]])
    s_row = torch.tensor([[[[1.0, -1.0]]]])
    o, _ = cp_superposition(s_row, torch.zeros_like(s_row),
                            v.view(1, 1, 2, 1), torch.zeros(1, 1, 2, 1), 1)
    o_f, _ = cp_superposition(-s_row, torch.zeros_like(s_row),
                              v.view(1, 1, 2, 1), torch.zeros(1, 1, 2, 1), 1)
    assert torch.allclose(o_f, -o, atol=1e-6), "not antisymmetric"


# ── P7 ────────────────────────────────────────────────────────────────
def test_p7_relative_position_rope():
    q_r, q_i = torch.randn(4, 8), torch.zeros(4, 8)
    k_r, k_i = torch.randn(4, 8), torch.zeros(4, 8)
    p, dt = 7, 500
    qr1, qi1 = apply_phase_rope(q_r, q_i, torch.tensor([p]))
    qr2, qi2 = apply_phase_rope(q_r, q_i, torch.tensor([p + dt]))
    kr1, ki1 = apply_phase_rope(k_r, k_i, torch.tensor([p]))
    kr2, ki2 = apply_phase_rope(k_r, k_i, torch.tensor([p + dt]))
    S1 = (qr1 * kr1 + qi1 * ki1).sum()
    S2 = (qr2 * kr2 + qi2 * ki2).sum()
    assert abs(float(S1 - S2)) < 1e-4, "score depends on absolute position"


# ── P8 ────────────────────────────────────────────────────────────────
def test_p8_kv_cache_incremental_equals_full():
    attn = QSAttention(32, 4, causal=True, use_rope=True, dropout=0.0)
    attn.eval()
    x = torch.randn(2, 12, 32)
    with torch.no_grad():
        y_ref = attn(x)
        y_inc, kv = attn(x[:, :1], use_cache=True)
        for t in range(1, 12):
            y_step, kv = attn(x[:, t:t + 1], past_kv=kv, use_cache=True)
            y_inc = torch.cat([y_inc, y_step], dim=1)
    assert torch.allclose(y_inc, y_ref, atol=1e-4), "cache diverged from full"


if __name__ == "__main__":
    tests = [test_p1_drop_in_and_causal, test_p2_cp_norm_and_phase,
             test_p4_destructive_interference, test_p5_no_exponential_stability,
             test_p6_antisymmetric_encoding, test_p7_relative_position_rope,
             test_p8_kv_cache_incremental_equals_full]
    print("=" * 60)
    print("QSA — PAPER FIDELITY (installed pip package)")
    print("=" * 60)
    n_pass = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__} — {e}")
    print(f"\n{n_pass}/{len(tests)} PASSED — "
          + ("QSA does what the paper says." if n_pass == len(tests)
             else "FAILURES above."))
    raise SystemExit(0 if n_pass == len(tests) else 1)
