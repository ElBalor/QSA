import math
import torch
import torch.nn as nn

from qsa import (
    ComplexLinear,
    QSAttention,
    ComplexLayerNorm,
    QSDecoderBlock,
    LogBornHead,
    ComplexClassifierHead,
    ExpandedComplexLinear,
    spectral_clamp,
    zero_frozen_gradients,
)

torch.manual_seed(0)


def test_complex_linear_matches_manual_complex_multiplication():
    layer = ComplexLinear(4, 3)
    x_r = torch.randn(2, 5, 4)
    x_i = torch.randn(2, 5, 4)
    y_r, y_i = layer(x_r, x_i)

    W_r = layer.fc_r.weight
    W_i = layer.fc_i.weight
    expect_r = x_r @ W_r.t() - x_i @ W_i.t()
    expect_i = x_r @ W_i.t() + x_i @ W_r.t()

    assert torch.allclose(y_r, expect_r, atol=1e-6)
    assert torch.allclose(y_i, expect_i, atol=1e-6)


def test_qsa_output_shape_and_finite():
    attn = QSAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 16, 32)
    out = attn(x)
    assert out.shape == (2, 16, 32)
    assert torch.isfinite(out).all()


def test_qsa_causal_masking():
    """Position t must not attend to positions > t: permuting the future must
    not change any output up to and including the last position of the prefix."""
    attn = QSAttention(d_model=32, n_heads=4, causal=True)
    attn.eval()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        out1 = attn(x)
        x2 = x.clone()
        x2[0, 5:] = torch.randn(3, 32)  # perturb the future
        out2 = attn(x2)
    # Positions 0..4 (and their residual) unchanged; position 5+ may change.
    assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-5)
    assert not torch.allclose(out1[0, 5:], out2[0, 5:], atol=1e-5)


def test_qsa_padding_mask():
    attn = QSAttention(d_model=32, n_heads=4, causal=False)
    attn.eval()
    x = torch.randn(1, 6, 32)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        full = attn(x, attention_mask=mask)
        # Truncating to the non-padded prefix must give identical outputs
        trimmed = attn(x[:, :4], attention_mask=mask[:, :4])
    assert torch.allclose(full[0, :4], trimmed[0, :4], atol=1e-5)


def test_qsa_is_subtractive():
    """The core claim: destructive interference is native in CP superposition.

    Hand-constructed, exact — no training. One token pair with equal scores:
    * aligned phases (S = 1 + 0j for both) -> amplitudes ADD (constructive);
    * opposed phases (S = 1 vs S = -1)     -> amplitudes CANCEL (destructive).
    A softmax attention output is a convex combination of values, so with equal
    scores it must average the two values in both cases. QSA does not.
    """
    from qsa import cp_superposition

    d_k = 4
    v_r = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    v_i = torch.zeros(2, d_k)

    # Constructive: both scores phase-aligned -> amplitudes add.
    # Row-norm alpha gives sum of |alpha| = sqrt(K)/sqrt(d_k) for K aligned keys.
    S_r_c = torch.ones(1, 2)
    S_i_c = torch.zeros(1, 2)
    out_r, out_i = cp_superposition(S_r_c, S_i_c, v_r, v_i, d_k)
    expected = torch.zeros(1, d_k)
    expected[0, 0] = math.sqrt(2.0 / d_k)  # values are nonzero only in dim 0
    assert torch.allclose(out_r, expected, atol=1e-5)
    assert torch.allclose(out_i, torch.zeros(1, d_k), atol=1e-5)

    # Destructive: scores 1 and -1 (pi phase offset) -> exact cancellation
    S_r_d = torch.tensor([[1.0, -1.0]])
    S_i_d = torch.zeros(1, 2)
    out_r, out_i = cp_superposition(S_r_d, S_i_d, v_r, v_i, d_k)
    assert torch.allclose(out_r, torch.zeros(1, d_k), atol=1e-5)
    assert torch.allclose(out_i, torch.zeros(1, d_k), atol=1e-5)

    # The same score *magnitudes* under softmax: a convex combination of equal
    # values is always the value itself — softmax can never output zero here.
    a = torch.softmax(S_r_d / math.sqrt(d_k), dim=-1)
    softmax_out = a @ v_r
    expected_softmax = torch.zeros(1, d_k)
    expected_softmax[0, 0] = 1.0  # w1*1 + w2*1 = 1 for ANY weights; dims 1-3 stay 0
    assert torch.allclose(softmax_out, expected_softmax, atol=1e-5), (
        "softmax always averages with non-negative weights; it cannot cancel"
    )


def test_decoder_block_backward_and_finite():
    block = QSDecoderBlock(d_model=16, n_heads=4, d_ff=32)
    r = torch.randn(2, 10, 16, requires_grad=True)
    i = torch.randn(2, 10, 16)
    out_r, out_i = block(r, i)
    assert out_r.shape == r.shape and out_i.shape == i.shape
    assert torch.isfinite(out_r).all() and torch.isfinite(out_i).all()
    out_r.sum().backward()
    assert r.grad is not None


def test_log_born_head_clamps():
    head = LogBornHead(d_model=8, vocab_size=10)
    r = torch.full((1, 4, 8), 1e4)
    i = torch.zeros(1, 4, 8)
    logits = head(r, i)
    assert logits.max().item() <= 50.0 and logits.min().item() >= -50.0


def test_classifier_head_pooled_mean_with_mask():
    head = ComplexClassifierHead(8, 2)
    r = torch.randn(2, 6, 8)
    i = torch.randn(2, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]])
    logits = head(r, i, attention_mask=mask)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_expanded_complex_linear_starts_at_zero_delta():
    base = ComplexLinear(8, 8)
    graft = ExpandedComplexLinear(base, expansion_dim=16)
    x_r = torch.randn(2, 8)
    x_i = torch.randn(2, 8)
    with torch.no_grad():
        base_r, base_i = base(x_r, x_i)
        g_r, g_i = graft(x_r, x_i)
    assert torch.allclose(base_r, g_r, atol=1e-7)
    assert torch.allclose(base_i, g_i, atol=1e-7)


def test_expanded_complex_linear_merge_zero_overhead():
    base = ComplexLinear(8, 8)
    # Linear variant: merge must be exact
    graft = ExpandedComplexLinear(base, expansion_dim=16, rms_scaling=False)
    # Train the expansion a little
    x_r, x_i = torch.randn(4, 8), torch.randn(4, 8)
    opt = torch.optim.Adam([p for p in graft.parameters() if p.requires_grad], lr=0.01)
    for _ in range(5):
        opt.zero_grad()
        r, i = graft(x_r, x_i)
        loss = r.pow(2).mean() + i.pow(2).mean()
        loss.backward()
        opt.step()

    before = graft(x_r, x_i)
    merged = graft.merge()
    after = merged(x_r, x_i)
    assert torch.allclose(before[0], after[0], atol=1e-5)
    assert torch.allclose(before[1], after[1], atol=1e-5)


def test_expanded_complex_linear_rms_variant_defaults():
    """Paper default (RMS-scaled, zero-init) must match the frozen base at init."""
    base = ComplexLinear(8, 8)
    graft = ExpandedComplexLinear(base, expansion_dim=16)  # rms_scaling=True default
    x_r = torch.randn(2, 8)
    x_i = torch.randn(2, 8)
    with torch.no_grad():
        base_r, base_i = base(x_r, x_i)
        g_r, g_i = graft(x_r, x_i)
    assert torch.allclose(base_r, g_r, atol=1e-6)
    assert torch.allclose(base_i, g_i, atol=1e-6)


def test_spectral_clamp_bounds_sigma():
    layer = ComplexLinear(64, 64)
    with torch.no_grad():
        layer.fc_r.weight.mul_(50.0)  # blow up the spectral norm
        layer.fc_i.weight.mul_(50.0)
    spectral_clamp(layer, gamma=1.0, max_iter=50)
    for fc in (layer.fc_r, layer.fc_i):
        w = fc.weight.data
        u = torch.randn(w.size(0))
        v = torch.randn(w.size(1))
        for _ in range(100):
            v = torch.mv(w.t(), u)
            v /= (torch.norm(v) + 1e-12)
            u = torch.mv(w, v)
            u_norm = torch.norm(u)
            u /= (u_norm + 1e-12)
        assert u_norm <= 1.0 + 0.05, f"spectral norm {u_norm} exceeds gamma"


def test_zero_frozen_gradients_safety_net():
    base = ComplexLinear(8, 8)
    graft = ExpandedComplexLinear(base, 8)
    # Unfreeze the base manually — the safety net should zero its grads
    for p in graft.base.parameters():
        p.requires_grad = True
    x = (torch.randn(4, 8), torch.randn(4, 8))
    r, i = graft(*x)
    (r.sum() + i.sum()).backward()
    zero_frozen_gradients(graft)
    assert torch.count_nonzero(graft.base.fc_r.weight.grad) == 0
    assert torch.count_nonzero(graft.base.fc_i.weight.grad) == 0


def test_spectral_clamp_after_step_training_stays_finite():
    """Deep QSA stack with the paper's stability regimen trains without NaN."""
    torch.manual_seed(1)
    blocks = nn.Sequential(*[QSDecoderBlock(16, 4, 32) for _ in range(8)])
    opt = torch.optim.AdamW(blocks.parameters(), lr=1e-3)
    x = torch.randn(2, 12, 16)
    i_stream = torch.zeros_like(x)
    for step in range(10):
        opt.zero_grad()
        r, i = x, i_stream
        for block in blocks:
            r, i = block(r, i)
        loss = r.pow(2).mean() + i.pow(2).mean()
        loss.backward()
        opt.step()
        spectral_clamp(blocks, gamma=1.0)
        assert torch.isfinite(loss).all(), f"loss went non-finite at step {step}"
